"""Detached GUI process; keep its lock until the window really closes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import traceback

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0,str(APP_DIR))
from instance_lock import FileLock
from bootstrap import atomic_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--token',required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{32}',args.token): return 1
    root = APP_DIR.parent; data = APP_DIR/'data'; data.mkdir(parents=True,exist_ok=True)
    ready = data/('.launch_'+args.token+'.json')
    lock = FileLock(data/'.app.lock')
    if not lock.acquire():
        atomic_json(ready,{'status':'already_running'}); return 0
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    path = logs/('app_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt')
    original_stdout,original_stderr=sys.stdout,sys.stderr
    with path.open('a',encoding='utf-8') as log:
        sys.stdout=sys.stderr=log
        try:
            atomic_json(data/'app_state.json',{'pid':os.getpid()})
            import start_scratch_app
            start_scratch_app.main(on_ready=lambda:atomic_json(ready,{'status':'ready','pid':os.getpid()}))
            return 0
        except Exception as exc:
            traceback.print_exc(file=log); log.flush()
            atomic_json(ready,{'status':'failed','error':type(exc).__name__})
            return 1
        finally:
            lock.release()
            sys.stdout,sys.stderr=original_stdout,original_stderr


if __name__=='__main__': sys.exit(main())
