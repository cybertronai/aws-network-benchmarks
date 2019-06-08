import base64
import os
import pickle
import subprocess
import shlex
import sys
import tempfile
import threading
from typing import List


class FileLogger:
    """Helper class to log to file (possibly mirroring to stderr)
     logger = FileLogger('somefile.txt')
     logger = FileLogger('somefile.txt', mirror=True)
     logger('somemessage')
     logger('somemessage: %s %.2f', 'value', 2.5)
  """

    def __init__(self, fn, mirror=True):
        self.fn = fn
        self.f = open(fn, 'w')
        self.mirror = mirror
        print(f"Creating FileLogger on {os.path.abspath(fn)}")

    def __call__(self, s='', *args):
        """Either ('asdf %f', 5) or (val1, val2, val3, ...)"""
        if (isinstance(s, str) or isinstance(s, bytes)) and '%' in s:
            formatted_s = s % args
        else:
            toks = [s] + list(args)
            formatted_s = ', '.join(str(s) for s in toks)

        self.f.write(formatted_s + '\n')
        self.f.flush()
        if self.mirror:
            # use manual flushing because "|" makes output 4k buffered instead of
            # line-buffered
            sys.stdout.write(formatted_s + '\n')
            sys.stdout.flush()

    def __del__(self):
        self.f.close()


def ossystem(cmd):
    """Like os.system, but returns output of command as string."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    (stdout, stderr) = p.communicate()
    return stdout.decode('ascii')


def get_global_rank():
    return int(os.environ['RANK'])


def get_world_size():
    return int(os.environ['WORLD_SIZE'])


def network_bytes():
    """Returns received bytes, transmitted bytes.

  Adds up recv/transmit bytes over all interfaces from /dev/net output which looks like this

Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 22127675  382672    0    0    0     0          0         0 22127675  382672    0    0    0     0       0          0
  ens5: 359138188558 51325343    0    0    0     0          0         0 363408452166 51916466    0    0    0     0       0          0
docker0:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0

"""

    proc = subprocess.Popen(['cat', '/proc/net/dev'], stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('ascii')

    recv_bytes = 0
    transmit_bytes = 0
    lines = stdout.strip().split('\n')
    lines = lines[2:]  # strip header
    for line in lines:
        line = line.strip()
        # ignore loopback interface
        if line.startswith('lo'):
            continue
        toks = line.split()

        recv_bytes += int(toks[1])
        transmit_bytes += int(toks[9])
    return recv_bytes, transmit_bytes


def parallelize(f, xs: List) -> None:
    """Executes f over all entry in xs in parallel, if any threads raise exceptions, propagate the first one."""

    exceptions = []

    def f_wrapper(x_):
        try:
            f(x_)
        except Exception as e:
            exceptions.append(e)

    threads = []
    for i, x in enumerate(xs):
        threads.append(threading.Thread(name=f'parallelize_{i}',
                                        target=f_wrapper, args=[x]))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if exceptions:
        raise exceptions[0]


# this helper is here in case we later want to capture huge stderr that doesn't fit in RAM
class TemporaryFileHelper:
    """Provides a way to fetch contents of temporary file."""
    def __init__(self, temporary_file):
        self.temporary_file = temporary_file

    def getvalue(self) -> str:
        return open(self.temporary_file.name).read()


STDOUT = 1
STDERR = 2


class capture_stderr:
  """Utility to capture output, use as follows
     with util.capture_stderr() as stderr:
        sess = tf.Session()
    print("Captured:", stderr.getvalue()).
    """

  def __init__(self, fd=STDERR):
    self.fd = fd
    self.prevfd = None

  def __enter__(self):
    t = tempfile.NamedTemporaryFile()
    self.prevfd = os.dup(self.fd)
    os.dup2(t.fileno(), self.fd)
    return TemporaryFileHelper(t)

  def __exit__(self, exc_type, exc_value, traceback):
    os.dup2(self.prevfd, self.fd)


class capture_stdout:
  """Utility to capture output, use as follows
     with util.capture_stdout() as stdout:
        sess = tf.Session()
    print("Captured:", stdout.getvalue()).
    """

  def __init__(self, fd=STDOUT):
    self.fd = fd
    self.prevfd = None

  def __enter__(self):
    t = tempfile.NamedTemporaryFile()
    self.prevfd = os.dup(self.fd)
    os.dup2(t.fileno(), self.fd)
    return TemporaryFileHelper(t)

  def __exit__(self, exc_type, exc_value, traceback):
    os.dup2(self.prevfd, self.fd)


def text_pickle(obj) -> str:
    """Pickles object into character string"""
    pickle_string = pickle.dumps(obj)
    pickle_string_encoded: bytes = base64.b64encode(pickle_string)
    s = pickle_string_encoded.decode('ascii')
    return s


def text_unpickle(pickle_string_encoded: str):
    """Unpickles character string"""
    if not pickle_string_encoded:
        return None
    obj = pickle.loads(base64.b64decode(pickle_string_encoded))
    return obj
