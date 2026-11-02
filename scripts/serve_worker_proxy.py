"""CPU gateway for an OpenVA GPU worker; report offline workers honestly."""
import argparse
import asyncio
import html
import json
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web


DROP = {'host', 'connection', 'upgrade', 'transfer-encoding', 'keep-alive',
        'proxy-authenticate', 'proxy-authorization', 'te', 'trailer',
        'cookie', 'authorization'}


def unavailable(request, reason, message):
    """Keep health/media failures non-200; give navigations a readable page."""
    payload = {'error': 'OpenVA worker unavailable', 'status': 'unavailable',
               'reason': reason, 'message': message}
    headers = {'Cache-Control': 'no-store', 'Retry-After': '30'}
    if request.path == '/' and 'text/html' in request.headers.get('Accept', ''):
        return web.Response(status=503, content_type='text/html', headers=headers,
                            text='<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                            '<meta name="viewport" content="width=device-width">'
                            '<title>OpenVA 后端离线</title><body style="background:#0b0e17;'
                            'color:#e5e7eb;font:18px/1.8 system-ui;padding:8vw">'
                            '<h1>OpenVA GPU 后端暂未运行</h1><p>' + html.escape(message) +
                            '</p><p>页面入口可达，但现在无法生成声音和人像。'
                            '后端恢复后请重新加载此页。</p></body></html>')
    return web.json_response(payload, status=503, headers=headers)


def create_app(upstream, availability_file=None):
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def context(app):
        async with ClientSession(auto_decompress=False,
                                 timeout=ClientTimeout(total=None, sock_connect=8)) as session:
            app['session'] = session
            yield

    async def forward(request):
        target = upstream
        # This operator-owned file is updated atomically after verifying the job.
        # It is not a health check: a ready record still must reach the real worker.
        if availability_file is not None:
            try:
                state = json.loads(Path(availability_file).read_text(encoding='utf-8'))
                if state['status'] != 'ready':
                    return unavailable(request, state['reason'], state['message'])
                target = state['upstream']
                if not isinstance(target, str) or not target.startswith(('http://', 'https://')):
                    raise ValueError('invalid worker URL')
            except (OSError, ValueError, KeyError, TypeError):
                return unavailable(request, 'configuration_unavailable',
                                   '后端配置尚未完成核验，请等待服务维护完成。')
        url = target.rstrip('/') + str(request.rel_url)
        session = request.app['session']
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            try:
                worker = await session.ws_connect(url, max_msg_size=0, heartbeat=30)
            except (ClientError, OSError, asyncio.TimeoutError, ValueError):
                return unavailable(request, 'worker_unreachable', '无法连接 GPU 后端，尚未建立媒体连接。')
            client = web.WebSocketResponse(max_msg_size=0, heartbeat=30)
            await client.prepare(request)

            async def pump(source, target):
                async for message in source:
                    if message.type == WSMsgType.TEXT:
                        await target.send_str(message.data)
                    elif message.type == WSMsgType.BINARY:
                        await target.send_bytes(message.data)
                    elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break

            tasks = [asyncio.create_task(pump(worker, client)),
                     asyncio.create_task(pump(client, worker))]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await worker.close()
                await client.close()
            return client
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
        try:
            response = await session.request(request.method, url, data=await request.read(),
                                             headers=headers, allow_redirects=False)
        except (ClientError, OSError, asyncio.TimeoutError, ValueError):
            return unavailable(request, 'worker_unreachable', '无法连接 GPU 后端，请检查推理作业状态。')
        async with response:
            forwarded = web.StreamResponse(status=response.status, headers={
                k: v for k, v in response.headers.items() if k.lower() not in DROP})
            await forwarded.prepare(request)
            async for chunk in response.content.iter_chunked(65536):
                await forwarded.write(chunk)
            await forwarded.write_eof()
            return forwarded

    app.cleanup_ctx.append(context)
    app.router.add_route('*', '/{path:.*}', forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--port', type=int, default=18786)
    parser.add_argument('--availability-file', type=Path)
    args = parser.parse_args()
    web.run_app(create_app(args.upstream, args.availability_file), host='0.0.0.0',
                port=args.port, access_log=None)


if __name__ == '__main__':
    main()
