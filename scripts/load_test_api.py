"""手动小规模真实 API 负载检查；仅测试自建会话，不重试模型请求。"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stages', default='1,4,8')
parser.add_argument('--long-reply', action='store_true')
args = parser.parse_args()
stages = [int(value) for value in args.stages.split(',')]
if not stages or any(not 1 <= value <= 128 for value in stages) or sum(stages) > 256:
    parser.error('Each stage must be 1..128 and total requests <=256')
BASE = 'http://127.0.0.1:8000'
RUN = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
REPORT = Path('evaluation/reports') / ('concurrency-' + RUN + '.json')
REPORT.parent.mkdir(parents=True, exist_ok=True)
report = {'run_id': RUN, 'mode': 'chat', 'long_reply': args.long_reply, 'stages': [], 'sessions': [], 'cleanup': []}

def request(path, body=None, method=None, timeout=180):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {'code': 'non_json_error'}
        return exc.code, payload
    except Exception as exc:
        return 0, {'code': type(exc).__name__}

def save():
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

def ask(sid, token, gate):
    gate.wait()
    started = time.monotonic()
    status, data = request('/sessions/' + sid + '/messages', {
        'response_mode': 'chat', 'question': ('用80到120个中文字解释Python装饰器，结尾单独写出本次测试标记：' if args.long_reply else '只回复以下测试标记，不要解释：') + token,
    })
    trace = data.get('trace') or {}
    events = trace.get('events', [])
    return {'session_id': sid, 'http_status': status, 'status': data.get('status'),
            'code': data.get('code'), 'stop_reason': data.get('stop_reason'),
            'seconds': round(time.monotonic()-started, 3),
            'marker_correct': token in data.get('answer', ''),
            'model_calls': data.get('model_calls'), 'tool_calls': data.get('tool_calls'),
            'model_ms': [e.get('duration_ms') for e in events if e.get('stage') == 'model']}

report['initial'] = request('/concurrency', timeout=5)[1]
try:
    for count in stages:
        sessions = []
        for i in range(count):
            status, data = request('/sessions', {'title': 'LOADTEST ' + RUN + ' ' + str(count) + '-' + str(i)}, timeout=15)
            if status != 201:
                raise RuntimeError('session_create_failed_' + str(status))
            sessions.append(data['session_id'])
            report['sessions'].append(data['session_id'])
            save()
        stage = {'concurrency': count, 'samples': [], 'results': []}
        report['stages'].append(stage)
        print('START stage=' + str(count), flush=True)
        gate = threading.Barrier(count + 1)
        with cf.ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(ask, sid, 'CHECK-' + RUN[-6:] + '-' + str(i), gate) for i, sid in enumerate(sessions)]
            start = time.monotonic()
            gate.wait()
            while not all(f.done() for f in futures):
                status, state = request('/concurrency', timeout=3)
                hstart = time.monotonic()
                hstatus, _ = request('/health', timeout=3)
                stage['samples'].append({'elapsed': round(time.monotonic()-start, 3), 'http': status,
                    'running': state.get('running'), 'waiting': state.get('waiting'),
                    'outstanding': state.get('outstanding'), 'health_status': hstatus,
                    'health_ms': round((time.monotonic()-hstart)*1000, 1)})
                time.sleep(0.2)
            stage['results'] = [f.result() for f in futures]
            stage['wall_seconds'] = round(time.monotonic()-start, 3)
        times = sorted(r['seconds'] for r in stage['results'])
        stage['summary'] = {'success': sum(r['http_status']==200 and r['status']=='success' for r in stage['results']),
            'markers_correct': sum(r['marker_correct'] for r in stage['results']),
            'mean_seconds': round(sum(times)/len(times),3), 'p95_seconds': times[math.ceil(len(times)*0.95)-1],
            'max_running': max((s.get('running') or 0 for s in stage['samples']), default=0),
            'max_waiting': max((s.get('waiting') or 0 for s in stage['samples']), default=0)}
        save()
        print(json.dumps({'stage': count, **stage['summary']}, ensure_ascii=False), flush=True)
        if stage['summary']['success'] != count:
            print('STOP: unsuccessful requests; no further escalation', flush=True)
            break
except Exception as exc:
    report['runner_error'] = type(exc).__name__ + ': ' + str(exc)
    print(report['runner_error'], flush=True)
finally:
    status, state = request('/concurrency', timeout=5)
    report['final_before_cleanup'] = state
    if status == 200 and state.get('outstanding') == 0:
        for sid in report['sessions']:
            status, _ = request('/sessions/' + sid, method='DELETE', timeout=15)
            report['cleanup'].append({'session_id': sid, 'http_status': status})
    report['final'] = request('/concurrency', timeout=5)[1]
    save()
    print('REPORT ' + str(REPORT.resolve()), flush=True)
