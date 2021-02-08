import subprocess as sub
import multiprocessing as mp


#subprocess.run("python real_time.py & python real_time_exit.py", shell=True)
def enter(q):
    sub.run(q)
def exit(w):
    sub.run(w)
if __name__=='__main__':
    en = mp.Process(target=enter,args=(["python","real_time.py"],))
    ex = mp.Process(target=exit,args=(["python","real_time_exit.py"],))
    en.start()
    ex.start()
    en.join()
    ex.join()