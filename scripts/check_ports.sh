#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import socket,sys
ports=(3600,8600,8610,8611)
busy=[]
for port in ports:
 s=socket.socket();
 try: s.bind(("127.0.0.1",port))
 except OSError: busy.append(port)
 finally: s.close()
if busy: print("ERROR: fixed ports are occupied: "+", ".join(map(str,busy)),file=sys.stderr); sys.exit(1)
print("ModelDeck fixed ports are available: "+", ".join(map(str,ports)))'

