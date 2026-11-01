"""Exercise the production clock branch under alternating pipeline backpressure."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def production_clock():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'server_mse.py').read_text(encoding='utf-8'))
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_feed_and_collect_loop')
    branch=next(n for n in ast.walk(method) if isinstance(n,ast.If) and ast.unparse(n.test)=='chunk is not None'
                and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='next_feed_time' for x in n.body))
    first=next(i for i,x in enumerate(branch.body) if isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='next_feed_time')
    branch.body=branch.body[first:]
    wrapper=ast.parse('def tick(next_feed_time, now, chunk, self):\n pass\n return next_feed_time\n')
    wrapper.body[0].body[0]=branch
    namespace={};exec(compile(ast.fix_missing_locations(wrapper),'<production feed clock>','exec'),namespace)
    return namespace['tick']


class FeedCadenceTests(unittest.TestCase):
    def test_backpressure_jitter_does_not_reduce_sustained_media_rate(self):
        tick=production_clock();engine=SimpleNamespace(args=SimpleNamespace(hop_ms=200))
        deadline=0.;ready=0.;submitted=[]
        for i in range(12000):
            now=i*.005
            if now+1e-9<deadline:continue
            available=now+1e-9>=ready
            deadline=tick(deadline,now,object() if available else None,engine)
            if available:
                submitted.append(now)
                ready=now+(.23 if len(submitted)%2 else .17)
        # Average model throughput is 25fps. Restarting the clock on every
        # blocked poll loses the 30ms debt each pair, producing only ~23fps.
        self.assertGreaterEqual(len(submitted)*5,1475)
        self.assertLessEqual(len(submitted)*5,1505)

    def test_retry_preserves_deadline_and_long_stall_has_bounded_catchup(self):
        tick=production_clock();engine=SimpleNamespace(args=SimpleNamespace(hop_ms=200))
        self.assertEqual(tick(.2,.23,None,engine),.2)
        self.assertEqual(tick(.2,10.,object(),engine),10.)
