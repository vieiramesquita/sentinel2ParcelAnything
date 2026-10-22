import shutil
import os

win_cred = "/mnt/c/Users/vinicius.mesquita/.config/earthengine/credentials"
wsl_cred = os.path.expanduser("~/.config/earthengine/credentials")

if os.path.exists(win_cred):
    shutil.copyfile(win_cred, wsl_cred)
    print("Credentials copied from Windows to WSL!")

import ee
try:
    ee.Initialize(project='ee-vieiramesquita')
    print("EE SUCCESS WITH PROJECT ee-vieiramesquita!")
except Exception as e:
    print("EE error with ee-vieiramesquita:", e)

try:
    ee.Initialize()
    print("EE SUCCESS WITHOUT PROJECT!")
except Exception as e:
    print("EE error default:", e)
