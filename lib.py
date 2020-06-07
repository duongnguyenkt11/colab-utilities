################################################################################
#                       Some inline utility functions                          #
################################################################################

def exec(code="ls -lah /content", result=True, verbose=False):
  from subprocess import check_output
  res = check_output(code.split(' '), universal_newlines=True)
  if verbose: print(res)
  if result: return res

FP_ENVIRON ='/content/environ.pickle'

def pickle_environ():
    import os, pickle, sys
    environ = os.environ
    dic = {key: environ[key] for key in os.environ.keys()}
    with open(FP_ENVIRON, 'wb') as file: pickle.dump(dic, file)
    with open(FP_ENVIRON+"2", 'wb') as file: pickle.dump(sys.path, file)

def load_environ():
    import os, pickle, sys
    with open(FP_ENVIRON, 'rb') as file: os.environ = pickle.load(file)
    with open(FP_ENVIRON+"2", 'rb') as file: sys.path = pickle.load(file)

def mmap(*args):
    return list(map(*args))

def num_cpus():
    from subprocess import call
    LOG_PATH = '/tmp/log.txt'
    with open(LOG_PATH, 'w') as file:
        call('cat /proc/cpuinfo | grep "model name" | wc ', shell=True, stdout=file)
    count = exec(f"cat {LOG_PATH}").split(' ')[5]
    return int(count)

def bench_mark_cpu(num_threads=40):
  exec('apt-get install sysbench')
  res = exec(f'sysbench --num-threads={num_threads} --test=cpu --cpu-max-prime=100000 run',
                result=True, verbose=False)
  # res = res.split('\n')[0]
  # return float(res.split(' ')[-1])
  return res

def port():
  return int(exec('allngrok', result=True).split(' ')[0].split(':')[-1])

def random_passwd():
    x = list("""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    from random import randint
    passwd =""
    for i in range(0, randint(20,30)): passwd += x[randint(0, len(x) - 1)]
    return passwd

def folder_exists(dir_path='/content/colab-utilities'):
  from os import path
  return path.exists(dir_path)

def exec2(cmd="curl -s http://localhost:4040/api/tunnels"): #exec2
  from subprocess import Popen, PIPE
  p = Popen(cmd.split(' '), stdout=PIPE)
  arr = p.communicate()
  res = arr[0].decode('utf-8')
  if len(res) >= 1 and res[-1]=="\n": return res[:-1]
  return res

def all_ngrok(port=4040, identifier="tcp://0.tcp.ngrok.io:", verbose=True):  #all_ngrok
  import json; from time import sleep; sleep(0.5)
  st = exec2(f"curl -s http://localhost:{port}/api/tunnels")
  lst = []
  while len(st) > 0:
    loc = st.find(identifier)
    if verbose:
      j = json.loads(st)
      print((j['tunnels'][0]['config']['addr'],j['tunnels'][0]['public_url']))
    lst.append(int(st[loc + len(identifier):loc+st[loc:].find('"')]))
    port += 1
    st = exec2(f"curl -s http://localhost:{port}/api/tunnels")
  return lst

#-------------------------------------------------------------------------------
#%%
