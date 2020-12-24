"""Get basic info of pcap


Split pcap:
    editcap -A “2017-07-07 9:00” -B ”2017-07-07 12:00” Friday-WorkingHours.pcap Friday-WorkingHours_09_00-12_00.pcap
    editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap
    # only save the first 10000 packets
    editcap -r AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap 0-10000

filter:
   cmd = f"tshark -r {input_file} -w {out_file} {srcIP_str}"
"""
import os
import subprocess
import numpy as np
import pickle

from itod_kjl.utils.utils import data_info, dump_data
from itod.data.pcap import PCAP, pcap2flows, flows2subflows


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data


def keep_ip(input_file, out_file='', kept_ips=['']):
    if out_file == '':
        ips_str = '-'.join(kept_ips)
        out_file = os.path.splitext(input_file)[0] + f'-src_{ips_str}.pcap'  # Split a path in root and extension.

    print(out_file)
    # only keep srcIPs' traffic
    srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
    cmd = f"tshark -r {input_file} -w {out_file} {srcIP_str}"

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def merge_pcaps(input_files, out_file):
    cmd = f"mergecap -w {out_file} " + ' '.join(input_files)

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def main():
    # input_file = 'data/data_reprst/pcaps/DEMO_IDS/DS-srcIP_192.168.10.5/Friday-WorkingHours/srcIP_192.168.10.5.pcap'

    ## wrccdc: https://archive.wrccdc.org/pcaps/2020/primary-site/feed3/
    # input_file = 'data/wrccdc/wrccdc.2020-03-20.174351000000002.pcap'

    ## DEFCON: https://oooverflow.io/dc-ctf-2018-finals/  (Network PCAPs are hosted by DEF CON)
    # input_file ='data/data_reprst/pcaps/DEFCON/ctf26/DEFCON26ctf_packet_captures-500000pkts.pcap'
    # input_file ='data/data_reprst/pcaps/DEFCON/ctf26/DEFCON26ctf_packet_captures.pcap'

    name = 'MACCDC'
    if 'ISTS' in name:
        # ## ISTS2015: https://www.netresec.com/?page=ISTS (https://download.netresec.com/pcap/ists-12/2015-03-07/)
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425741002.pcap'
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425741051.pcap'
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425823409.pcap'
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425842738.pcap'
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425824560.pcap'
        input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425824164.pcap'

        input_files = [
            'snort.log.1425741002.pcap',
            'snort.log.1425741051.pcap',
            'snort.log.1425823409.pcap',
            # 'snort.log.1425842738.pcap',
            # 'snort.log.1425824560.pcap',

        ]
        # input_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425824164.pcap' # for abnormal dataset
        input_files = [os.path.join('data/data_reprst/pcaps/ISTS/2015', v) for v in input_files]
        out_file = os.path.join('data/data_reprst/pcaps/ISTS/2015', 'snort.log-merged-3pcaps.pcap')
        merge_pcaps(input_files, out_file)

        # input_file = out_file

    elif 'MACCDC' in name:
        # maccdc: https://www.netresec.com/?page=MACCDC ()
        # input_file = 'data/data_reprst/pcaps/MACCDC2012/maccdc2012_00000-1000000pkts.pcap'
        # input_file = 'data/data_reprst/pcaps/MACCDC2012/maccdc2012_00000-src_192.168.229.153.pcap'
        input_file = 'data/data_reprst/pcaps/MACCDC/2012/maccdc2012_00000.pcap'

    elif 'MAWI' in name:

        # ## CIC_INVESAndMAL2019
        # input_file = 'data/data_reprst/pcaps/CIC_INVESAndMAL2019/Benign-5/06_16_2017-be-20170216-apps-a.envisionmobile.caa/06_16_2017-be-20170216-apps-a.envisionmobile.caa.pcap'

        # ## BROWSER_TRAFFIC: https://github.com/uoitdnalab/NetworkTrafficDataset
        # # BRAVE/ROUTE_TO__020fe964e3b0ccd96dbe7c57bfe97eb60be1648f7298cab698b3271a296c57b9
        # input_file ='data/data_reprst/pcaps/BROWSER_TRAFFIC/020fe964e3b0ccd96dbe7c57bfe97eb60be1648f7298cab698b3271a296c57b9.pcap'

        # ## ctu_iot23: Phillips_HUE
        # input_file ='data/data_reprst/pcaps/ctu_iot23/Phillips_HUE/2018-10-25-14-06-32-192.168.1.132.pcap'
        #
        # ## ctu_iot23: Soomfy_Doorlock
        # input_file = 'data/data_reprst/pcaps/ctu_iot23/Soomfy_Doorlock/2019-07-03-15-15-47-first_start_somfy_gateway.pcap'
        #
        # ## ctu_iot23: Amazon_Echo
        # input_file = 'data/data_reprst/pcaps/ctu_iot23/Amazon_Echo/2018-09-21-capture.pcap'

        # ##  ctu_iot23: CTU-IoT-Malware-Capture-7-1
        # input_file = 'data/data_reprst/pcaps/ctu_iot23/CTU-IoT-Malware-Capture-7-1/2018-07-20-17-31-20-192.168.100.108.pcap'

        ### MAWI:
        ## http://mawi.wide.ad.jp/mawi/samplepoint-F/2020/202007011400.html
        # input_file = 'data/data_reprst/pcaps/DS50_MAWI_WIDE/202007011400-10000000.pcap' # to get IP info first quickly.
        input_file = 'data/data_reprst/pcaps/DS50_MAWI_WIDE/202007011400.pcap'  # filter srcIP

    parse_flg = 0
    if parse_flg:
        flows, _ = pcap2flows(input_file)

        flows_info = {}
        for five_tuple, packet_times, packets in flows:
            src = five_tuple[0]
            if src not in flows_info.keys():
                flows_info[src] = 1
            else:
                flows_info[src] += 1

        sorted_flows_info = {k: v for k, v in sorted(flows_info.items(), key=lambda item: item[1], reverse=True) if
                             v > 50}
        print(sorted_flows_info)
        out_file = input_file + '-ips_stats.dat'
        dump_data(sorted_flows_info, out_file)
    out_file = input_file + '-ips_stats.dat'
    if os.path.exists(out_file):
        sorted_flows_info = load_data(input_file + '-ips_stats.dat')
        print(sorted_flows_info)

    filter_flg = 1
    if filter_flg:
        if 'WRCCDC' in input_file:
            # for wrccdc
            kept_ips = [['172.16.16.30'], ['172.16.16.16'], ['10.183.250.172']]

        if 'MACCDC' in input_file:
            # for maccdc
            kept_ips = [['192.168.202.79'], ['192.168.229.153'], ['192.168.202.76']]
            raw_out_file = os.path.splitext(input_file)[0] + '-srcIP_ips.pcap'
        elif 'DEFCON' in input_file:
            ## for DEFCON
            kept_ips = [['10.0.0.2'], ['10.13.37.23'], ['10.23.0.137'], ['10.21.0.139']]
        elif 'CTU_IOT23' in input_file:
            ## for ctu_iot23
            kept_ips = [['192.168.100.108'], ['46.28.110.244']]
        elif 'ISTS' in input_file:
            kept_ips = [[], ['10.2.4.30'], ['10.128.0.13', '10.0.1.51', '10.0.1.4', '10.2.12.40'], ['10.2.12.10'],
                        ['10.2.12.50'], ['10.2.12.60']]
            raw_out_file = os.path.join('data/data_reprst/pcaps/ISTS/2015', 'snort.log-merged-srcIP_ips.pcap')
        elif "MAWI" in input_file:
            # input_file = 'data/data_reprst/pcaps/DS50_MAWI_WIDE/202007011400-10000000.pcap'  # to get IP info first quickly.
            input_file = 'data/data_reprst/pcaps/DS50_MAWI_WIDE/202007011400.pcap'  # filter srcIP
            kept_ips = [['185.8.54.240'], ['203.78.7.165'], ['203.78.4.32'], ['222.117.214.171'], ['222.117.192.164'],
                        ['101.27.14.204'], ['103.114.160.226'], ['18.178.219.109'], ['14.208.165.132']]
            raw_out_file = 'data/data_reprst/pcaps/DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165/202007011400-srcIP_ips.pcap'

        for ips in kept_ips:
            if len(ips) > 0:
                # out_file = os.path.splitext(input_file)[0] + '-normal.pcap'
                # print(out_file.replace('ips', '-'.join(ips)))
                out_file = keep_ip(input_file, out_file=raw_out_file.replace('ips', '-'.join(ips)), kept_ips=ips)
            else:
                out_file = input_file
            # flows, _ = pcap2flows(out_file)
            # print(len(flows))

            # packet_times = [packet_times[-1]-packet_times[0] for flow, packet_times, packets in flows]
            # print(sorted(packet_times))
            # data_info(np.asarray(packet_times).reshape(-1, 1))
            # for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            #     print(f'q: {q}')
            #     interval=np.quantile(packet_times, q)
            #     subflows=flows2subflows(flows, interval=interval, num_pkt_thresh=2, data_name='', abnormal=False)
            #     print('=======', len(subflows))


if __name__ == '__main__':
    main()
