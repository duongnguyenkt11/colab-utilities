def tik(verbose=True):
    import time
    global _time
    _time = time. time()
    CEND = '\33[0m'; CVIOLET = '\33[35m'
    if verbose:
        print(f"Started timing: {CVIOLET}{time_format(time_only=True)}{CEND}.")


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
                print(f"{url:<40} =>   {local:<20} {port}")
        return True

    i = 4040
    while ngrok_info(i): i += 1

print_all()
"""

create_script('auth' , 'ngrok authtoken 12345')
create_script('tunnel' , 'ngrok authtoken 12345 && ngrok tcp 22 &')
create_script('files', 'ngrok http -subdomain=sotola file:///')
create_script('mybokeh', 'ngrok http -subdomain=sotobokeh 5100')
create_command('allngrok', code_all)

def my_time_stamp(t=-1, time_only=False, adjust = 7):
    import time
    if t == -1: t = time.time()
    st = time.strftime('%Y_%m_%d - %I_%M_%S %p',time.localtime(t + 3600* adjust))
    if time_only: st = st[-11:-3]
    return st

def execute(code, log_file=-1, verbose=True, result=False):
  import subprocess
  if log_file == -1: log_file = f"/tmp/logs/{my_time_stamp().split(' - ')[1]}-{code.split(' ')[0]}.txt"
  with open(log_file, "w+") as file:
    subprocess.call(code, shell=True, stdout=file)
  with open(log_file, "r") as file:
    res = file.read()
    if len(res) == 0: res = f"*** {code}: No output ***"
  if verbose: print(res)
  if result: return res
  else: return
    
#-----------------------------------------------------------------------------------------------------------

def start_colab_session(password, freetoken, token):
  import sys , time, os
  execute("mkdir /tmp/logs", "init.txt")
  print("*** Download ngrok ***")
  get_ipython().system_raw("tar -xvf /content/colab-utilities/pycharm_helper.tar.gz && mv .pycharm_helpers /root/")

  def make_yml_file(user='', authtoken=''):
    print("*** Preparing ngrok.yml file with ports for web-app fowarding ***")
    st = f"""
    authtoken: {authtoken} 
    tunnels:
      bokeh-app:
        addr: 5100
        proto: http
        subdomain: {user}-tpu-bokeh
      flask-app:
        addr: 5000
        proto: http
        subdomain: {user}-tpu-flask
      files-app:
        addr: file:///
        proto: http
        subdomain: {user}-tpu-files
      tensor-board:
        addr: 6006
        proto: http
        subdomain: {user}-tpu-tf
    """
    if user == 'dummy': st = f"""authtoken: {authtoken}"""
    with open('/root/.ngrok2/ngrok.yml', 'w') as file:
      file.write(st)
    time.sleep(0.2)
  # /make_yml_file()

  #app0
  execute("""wget -q -c -nc https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip""")
  execute("""unzip -qq -n ngrok-stable-linux-amd64.zip""")
  print("*** Listing current dir, which should contain ngrok ***")
  execute("ls")
  execute("""cp ./ngrok /usr/bin/""")

  execute("ngrok authtoken dummytoken") # create /root/.ngrok2/ folder
  make_yml_file(user='dummy', authtoken=freetoken) # tanky
  get_ipython().system_raw('ngrok tcp 22 &')
  make_yml_file(user='sotola', authtoken=token) 


  execute("cat /root/.ngrok2/ngrok.yml | grep authtoken")

  # Setup sshd
  print("*** Install ssh (using apt-get) ***")
  execute("""apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen > /dev/null""")
  #execute("""apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen""")
  time.sleep(3); print("*** Set root password ***")

  # Set root password 1
  execute("""mkdir /var/run/sshd""")
  execute(f"""echo root:{password} | chpasswd""")

  execute("""echo "PermitRootLogin yes" >> /etc/ssh/sshd_config """)
  execute("""echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config""")
  execute("""echo "LD_LIBRARY_PATH=/usr/lib64-nvidia" >> /root/.bashrc""")
  execute("""echo "export LD_LIBRARY_PATH" >> /root/.bashrc""")
  # Run sshd
  print("*** Starting ssh service ***")
  get_ipython().system_raw('/usr/sbin/sshd -D &')
  # Run sshd
  print("*** Starting ssh service ***")
  get_ipython().system_raw('/usr/sbin/sshd -D &')
  print("*** Checking for ssh daemon (sshd: /usr/sbin/sshd -D) ***")
  time.sleep(1)
  execute("""ps -aux | grep [s]sh""")
  print("*** checking pycharm_helper, there should be pycharm_test.py ***")
  execute("ls /root/.pycharm_helpers/pycharm/pycharm_commands")
  execute('allngrok')
  tok()


#-----------------------------------------------------------------------------------------------------------
tik()
