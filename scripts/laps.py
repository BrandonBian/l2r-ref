import re
import subprocess

if __name__ == '__main__':
    p = subprocess.Popen(['/bin/bash', 'laptimes.bash'],
                        stdout=subprocess.PIPE)

    print('Collecting laptimes')
    laptimes = []
    best_episode = 1e9
    best_lap = 1e9
    best_episode_laps = []
    ct = 0

    while p.poll() is None:
        l = str(p.stdout.readline())
        if len(l) < 5:
        	break
        raw = re.search(r'\[(.*?)\]', l).group(1)
        raw = raw.replace(',','')
        times = list(map(float, raw.split()))
        if len(times) == 3:
        	if sum(times) < best_episode:
        		best_episode = sum(times)
        		best_episode_laps = times

        best_lap = min(best_lap, min(times))
        for lt in times:
            if lt < 60:
                print(f'Wow! A lap time of {lt} in {times}')
                ct += 1

    print(f'Number of sub 60 laps: {ct}')
    print(f'Best lap: {best_lap:.2f}')
    print(f'Best episode: {best_episode:.2f}, {best_episode_laps}')

