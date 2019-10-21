def tik():
    import time
    global _time
    _time = time. time()

def tok():
    import time
    CEND = '\33[0m'; CVIOLET = '\33[35m'
    global _time
    print(f"Elapsed time: {CVIOLET}{round(time.time() - _time, 2)}{CEND} seconds. "
          f"({time_format(time_only=True)})")

def time_format(t=-1, time_only=False, adjust = 7):
    import time
    if t == -1: t = time.time()
    st = time.strftime('%Y/%m/%d  %I:%M:%S %p',time.localtime(t + 3600* adjust))
    if time_only: st = st[-11:]
    return st

# *** Creating scripts ***
print("*** Checking /tmp (should be empty) ***")

import time
time.sleep(0.1); print("*** Generate /usr/bin scripts ***")
output = get_ipython().system_raw

def func_wrapper(f, *args, delay=0.01):
    import threading
    return threading.Timer(delay, lambda: f(*args)).start

def create_script(name, code, shebang='#!/bin/bash'):
    output(f"""echo '{shebang}
    {code}' > /usr/bin/{name}
    chmod +x  /usr/bin/{name}
    """)

def create_command(name, code, shebang='#!/usr/local/bin/python', check=False):
    code = shebang + "\n" + code
    name = f'/usr/bin/{name}'
    with open(f'{name}', 'w') as file:
        file.write(code)
    output(f"""chmod +x  {name}""")
    if check: output(f"""cat  {name}""")


code_all = """# code start\n
def print_all():
    def ngrok_info(port=4040):
        import subprocess
        import json
        result = subprocess.run(['curl', "-s", f"http://localhost:{port}/api/tunnels"], stdout=subprocess.PIPE)
        if len(result.stdout) == 0: return False
        jobject = json.loads(result.stdout)
        for tunnel in jobject['tunnels']:
            url = tunnel['public_url']
            if url.count('https') <= 0 :
                local = tunnel['config']['addr'].replace('http://', '').replace('file://', '')
                print(f"{url:<30} =>   {local:<20} {port}")
        return True

    i = 4040
    while ngrok_info(i): i += 1

print_all()
"""

create_script('auth' , 'ngrok authtoken 1SNomSh04Xr4dDCzLinGGnUZ39A_3rKGLsQZG9qMkVzPz1Zv4')
create_script('tunnel' , 'ngrok authtoken 1S1bYIU5EsnjbF6yMmFUscJwGjj_5q3K8zNcMHSXXxHY3gB8B && ngrok tcp 22 &')
create_script('files', 'ngrok http -subdomain=sotola file:///')
create_script('mybokeh', 'ngrok http -subdomain=sotobokeh 5100')
create_command('allngrok', code_all)

tik()
