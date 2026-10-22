import urllib.request
import json
import time
import threading

def load_coords():
    req = urllib.request.Request(
        'http://127.0.0.1:8088/api/roi/coords',
        data=json.dumps({'lat': -17.55, 'lon': -51.25, 'buffer_m': 1500}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    urllib.request.urlopen(req)

t = threading.Thread(target=load_coords)
t.start()

# Monitor progress
for _ in range(25):
    time.sleep(0.2)
    res = urllib.request.urlopen('http://127.0.0.1:8088/api/progress')
    prog = json.loads(res.read())
    print(f"[{prog['percent']}%] {prog['title']} - {prog['message']}")
    if prog['percent'] == 100:
        break

t.join()
print("MONITORING SUCCESSFUL!")
