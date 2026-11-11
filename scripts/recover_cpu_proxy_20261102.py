"""One-time, guarded restart of our migrated OpenVA proxy on sunxc-v1 only.

Run through the existing notebook session. No login, GPU allocation, package
installation, or other process termination is performed by this script.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path('/inspire/ssd/project/video-generation/public/sunxincheng/openva_dialogue_20261030')
OUT = ROOT / 'availability_20261102'
OLD_PID = 142707
OLD_START_TICKS = '1364572794'
SCRIPT = ROOT / 'ops/dev_proxy.py'
PYTHON = ROOT / 'runtime/venvs/pipecat/bin/python'
UPSTREAM = 'http://10.246.1.141:18788'


def atomic_json(path, record):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    assert socket.gethostname().startswith('sunxc-v1--5fe56eb0405a-'), 'Wrong machine'
    old = Path('/proc') / str(OLD_PID)
    pidfd = os.pidfd_open(OLD_PID)
    try:
        assert (old / 'stat').read_text().split(') ', 1)[1].split()[19] == OLD_START_TICKS
        assert (old / 'cmdline').read_bytes().split(b'\0')[:-1] == [
            str(PYTHON).encode(), b'-B', str(SCRIPT).encode(),
            b'--upstream', UPSTREAM.encode(), b'--port', b'18786']
        assert (old / 'cwd').resolve() == ROOT
        # The source of this status is the platform receipt, not stale state.json.
        evidence = json.loads((OUT / 'platform_jobs_summary.json').read_text())
        assert evidence['old_status'] == 'job_stopped'
        state = {'status': 'unavailable', 'reason': 'gpu_stopped',
                 'message': '原 OpenVA GPU 作业已停止（2026-11-02 05:14:19）；'
                            '新 CPU 网页代理已运行，等待分配 OpenVA 专用 GPU。',
                 'job_id': evidence['old_job_id'], 'finished_at': evidence['finished_at'],
                 'checked_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(OUT / 'backend.json', state)
        backup = OUT / 'dev_proxy.before.py'
        if not backup.exists():
            backup.write_bytes(SCRIPT.read_bytes())
        # Preserve the original script entry used by the migration service inventory.
        wrapper = ('import runpy\nimport sys\n'
                   f'sys.argv.extend(["--availability-file", {str(OUT / "backend.json")!r}])\n'
                   f'runpy.run_path({str(ROOT / "repo/scripts/serve_worker_proxy.py")!r}, run_name="__main__")\n')
        staged = SCRIPT.with_suffix('.tmp')
        staged.write_text(wrapper)
        staged.replace(SCRIPT)
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        for _ in range(50):
            if not old.exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Old proxy did not exit; no replacement started')
    finally:
        os.close(pidfd)
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    with (OUT / 'proxy.log').open('ab') as log:
        process = subprocess.Popen([str(PYTHON), '-B', str(SCRIPT), '--upstream', UPSTREAM,
                                    '--port', '18786'], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = {'host': socket.gethostname(), 'pid': process.pid, 'replaced_pid': OLD_PID,
               'checked_at': datetime.now(timezone.utc).isoformat(), 'responses': {}}
    for route in ('/', '/health', '/ws/media'):
        for attempt in range(30):
            request = urllib.request.Request('http://127.0.0.1:18786' + route,
                                             headers={'Accept': 'text/html' if route == '/' else 'application/json'})
            try:
                response = urllib.request.urlopen(request, timeout=2)
            except urllib.error.HTTPError as error:
                response = error
            except OSError:
                if process.poll() is not None or attempt == 29:
                    raise
                time.sleep(0.1)
                continue
            with response:
                body = response.read().decode()
                assert response.code == 503, (route, response.code)
                assert ('GPU 后端暂未运行' in body if route == '/' else
                        json.loads(body)['reason'] == 'gpu_stopped')
                receipt['responses'][route] = {'http_status': response.code, 'body': body}
            break
    atomic_json(OUT / 'new_cpu_acceptance.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
