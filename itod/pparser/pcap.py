"""Preprocess pcap
Only keep given srcIPs' traffic, remove other traffic from a given pcap (too large)

Split pcap:
    editcap -A “2017-07-07 9:00” -B ”2017-07-07 12:00” Friday-WorkingHours.pcap Friday-WorkingHours_09_00-12_00.pcap
    editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap
    # only save the first 10000 packets
    editcap -r AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap 0-10000

Using tshark to filter packets
    : tshark -r {input_file} -w {output_file} ip.src=={srcIP}

For example,
 > tshark -r Friday-WorkingHours.pcap -w Friday-WorkingHours@5_Bots_SrcIPs-20170707-09_00-12_00.pcap
                              ip.src==192.168.10.5 or ip.src==192.168.10.8 or ip.src==192.168.10.9
                              or ip.src==192.168.10.14 or ip.src==192.168.10.15

 > tshark -r Monday-WorkingHours.pcap -w Monday-WorkingHours@5_SrcIPs-Normal.pcap
                              "ip.src==192.168.10.5 or ip.src==192.168.10.8 or ip.src==192.168.10.9
                              or ip.src==192.168.10.14 or ip.src==192.168.10.15"

Unfortunately the speed of vanilla Linux kernel networking is not sufficient for more specialized workloads.
For example, here at CloudFlare, we are constantly dealing with large packet floods.
Vanilla Linux can do only about 1M pps. This is not enough in our environment,
especially since the network cards are capable of handling a much higher throughput.
Modern 10Gbps NIC’s can usually process at least 10M pps.
Such as DPDK, Moongen
Ref: https://serverascode.com/2018/12/31/ten-million-packets-per-second.html

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
import datetime
import os.path as pth
from collections import Counter
from shutil import copyfile

from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP
from sklearn.utils import shuffle

from itod.utils.tool import *


class PCAP:
    """Pcap interface including filter_ip, save, and so on.

    """

    def __init__(self, pcap_file='', subflow_flg=False, num_pkt_thresh=2, subflow_interval=0.1,
                 q_flow_durations=0.5,
                 overwrite=True,
                 verbose=True,
                 **kwargs):
        self.pcap_file = pcap_file
        self.subflow_flg = subflow_flg
        self.num_pkt_thresh = num_pkt_thresh
        self.subflow_interval = subflow_interval
        self.q_flow_durations = q_flow_durations  # self.q_flow_durations = 0.5  # default 0.5
        self.overwrite = overwrite
        self.verbose = verbose

        if len(kwargs) > 0:
            for i, (key, value) in enumerate(kwargs.items()):
                setattr(self, key, value)

    def filter_ip(self, pcap_file, kept_ips=[], output_file=''):

        if output_file == '':
            output_file = os.path.splitext(pcap_file)[0] + 'filtered_ips.pcap'  # Split a path in root and extension.
        # only keep srcIPs' traffic
        srcIP_str = " or ".join([f'ip.src!={srcIP}' for srcIP in kept_ips])
        cmd = f"tshark -r {pcap_file} -w {output_file} {srcIP_str}"

        print(f'{cmd}')
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        return output_file

    def keep_ip(self, pcap_file, kept_ips=[], output_file=''):

        if output_file == '':
            output_file = os.path.splitext(pcap_file)[0] + 'kept_ips.pcap'  # Split a path in root and extension.
        # only keep srcIPs' traffic
        srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
        cmd = f"tshark -r {pcap_file} -w {output_file} {srcIP_str}"

        print(f'{cmd}')
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        return output_file

    def merge_pcaps(self, pcap_file_lst=[], output_file='merged.pcap'):
        num_pcap = len(pcap_file_lst)
        if num_pcap == 0:
            msg = f'{len(pcap_file_lst)} pcaps need to be merged.'
            raise ValueError(msg)
        else:

            if os.path.exists(output_file):
                if self.overwrite:  os.remove(output_file)
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            cmd = f"mergecap -w {output_file} " + ' '.join(pcap_file_lst)
            print(f'{cmd}')
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
            except Exception as e:
                print(f'{e}, {result}')
                return -1
        return output_file

    def stats_flows(self, flows=''):
        # fids, features = list(*zip(features))
        self.fids = list(map(lambda x: x[0], flows))
        self.features = list(map(lambda x: (x[1], x[2]), flows))  # times, pkts

        self.flows_durations = [float(pkt_times[-1]) - float(pkt_times[0]) for pkt_times, pkts in self.features]
        self.num_pkts_per_flow = [len(pkts) for pkt_times, pkts in self.features]
        self.num_pkts = sum(self.num_pkts_per_flow)

        if self.verbose:
            stat_data(np.asarray(self.flows_durations).reshape(-1, 1), name='flows_durations')
            print(f'median: {np.quantile(self.flows_durations, q=0.5)}, mean: {np.mean(self.flows_durations)}')
            stat_data(np.asarray(self.num_pkts_per_flow).reshape(-1, 1), name='num_pkts_per_flow')
            print(
                f'{self.num_pkts} TCP and UDP packets in \'{self.pcap_file}\', flows with less than 2 packets are filtered.')

    def read_labels(self):
        self.labels = []
        if hasattr(self, 'label_file'):
            with open(self.label_file, 'r') as f:
                line = f.readline()
                while line:
                    # Todo
                    line = f.readline()

    def label_flows(self, flows='', label_file=''):
        time_windows = label_file

        def time_in_time_window(pkt_times, time_window):  # judge if pkt_times in time_window)
            start_time, end_time = time_window
            if np.min(pkt_times) > str2time(end_time) or np.max(pkt_times) < str2time(start_time):
                return False
            else:
                return True

        self.labels = []
        for i, (fid, pkt_times, pkts) in enumerate(flows):
            in_time_flg = False
            for j, time_window in enumerate(time_windows):
                if time_in_time_window(pkt_times, time_window):
                    in_time_flg = True
                    break
            if in_time_flg:
                label_i = 'ANOMALY'
            else:
                label_i = 'NORMAL'
            self.labels.append(label_i)

        return self.labels

    def label_flows_with_times(self, flows='', time_windows=[('2019-12-10 13:26:00', '2019-12-10 13:27:00'),
                                                             ('start_time', 'end_time')]):

        def time_in_time_window(pkt_times, time_window):  # judge if pkt_times in time_window)
            start_time, end_time = time_window
            if np.min(pkt_times) > str2time(end_time) or np.max(pkt_times) < str2time(start_time):
                return False
            else:
                return True

        self.labels = []
        for i, (fid, pkt_times, pkts) in enumerate(flows):
            in_time_flg = False
            for j, time_window in enumerate(time_windows):
                if time_in_time_window(pkt_times, time_window):
                    in_time_flg = True
                    break
            if in_time_flg:
                label_i = 'ANOMALY'
            else:
                label_i = 'NORMAL'
            self.labels.append(label_i)

        return self.labels

    @func_notation
    @execute_time
    def pcap2flows_with_pcap_label(self, pcap_file, labels_csv, subflow=False,
                                   output_flows_labels='_flows_labels', label_file_type=''):
        if os.path.exists(output_flows_labels + '-all.dat'):
            flows, labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels + '-all.dat')
            return flows, labels

        # get all flows which at least has more than 2 packets
        self.subflow = subflow
        if self.subflow:
            print('---calculate subflow interval from normal pcap')
            flows_tmp, num_pkts = pcap2flows(pcap_file, self.num_pkt_thresh)
            print(f'num. of flows: {len(flows_tmp)}')
            new_fids, new_labels = _load_labels_and_label_flows(labels_csv,
                                                                features=[(fid, _) for fid, _, _ in flows_tmp],
                                                                label_file_type=label_file_type)
            # only get normal flows durations
            self.flows_durations = [max(pkt_times) - min(pkt_times) for (fids, pkt_times, pkts), label in
                                    zip(flows_tmp, new_labels) if label.upper() in ['NORMAL', 'BENIGN']]
            stat_data(np.asarray(self.flows_durations, dtype=float).reshape(-1, 1), name='flows_durations')

            self.subflow_interval = self.params['subflow_interval']
            if type(self.subflow_interval) == float and \
                    (self.subflow_interval > 0) and (self.subflow_interval < max(self.flows_durations)) \
                    and (self.subflow_interval != None):
                print(f'+++subflow_interval: ', self.subflow_interval, ', q_flows_durations:',
                      self.params['q_flow_dur'])
            else:
                # self.q_flow_dur = 0.5 # come from PCAP() default is 0.5
                print(
                    f'intervals: {np.quantile(self.flows_durations, q=[0.25, 0.5, 0.75, 0.9, 0.95])}, when q=[0.25, 0.5, 0.75, 0.9, 0.95]')
                self.q_flow_dur = self.params['q_flow_dur']
                self.subflow_interval = np.quantile(self.flows_durations,
                                                    q=self.q_flow_dur)  # median  of flow_durations
                print(f'---subflow_interval: ', self.subflow_interval, f', q_flow_dur: {self.q_flow_dur}')
                self.params['subflow_interval'] = self.subflow_interval
                # self.params['q_flow_dur'] = self.q_flow_dur
            flows = _load_pcap_to_subflows(pcap_file, num_pkt_thresh=self.num_pkt_thresh, max_pkts=-1,
                                           interval=self.subflow_interval)
            # print("Number of FIDs in each subflow: {}".format([len(fids) for fids in list(zip(*flows))[0]]))
            print(f'num. of flows after splitting with inteval: {len(flows)}')
            # labels = _load_labels_and_label_flows_by_data(list(zip(new_fids, new_labels)),
            #                                               features=[(fid, _) for fid, _, _ in flows],
            #                                               label_file_type=label_file_type)
            fids, labels = _load_labels_and_label_flows(labels_csv,
                                                        features=[(fid, _) for fid, _, _ in flows],
                                                        label_file_type=label_file_type)
            print(f'num. of flows after labeling with label_csv: {len(fids)}, Counter(labels): {Counter(labels)}')
        else:
            flows, num_pkts = pcap2flows(pcap_file,
                                         self.num_pkt_thresh)  # get all flows which at least has more than 2 packets
            print(f'num. of flows: {len(flows)}')
            fids, labels = _load_labels_and_label_flows(labels_csv, features=[(fid, _) for fid, _, _ in flows],
                                                        label_file_type=label_file_type)

        return flows, labels

    @func_notation
    @execute_time
    def pcap2flows_with_pcaps(self, pcap_file_lst, subflow=False, output_flows_labels='_flows_labels'):
        if os.path.exists(output_flows_labels + '-all.dat'):
            flows, labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels + '-all.dat')
            return flows, labels

        flows = []
        labels = []
        self.subflow = subflow
        for i, pcap_file in enumerate(pcap_file_lst):  # the first file must be 'normal.pcap'
            # get all flows which at least has more than 2 packets
            # flows_i = pcap2flows(pcap_file, self.num_pkt_thresh)
            if self.subflow:
                if i == 0:
                    print('---calculate subflow interval from normal pcap')
                    if 'normal.pcap' in pcap_file:  # only get subflow_interval from normal.pcap
                        flows_tmp, num_pkts = pcap2flows(pcap_file, self.num_pkt_thresh)  # all full flows
                        self.flows_durations = [max(pkt_times) - min(pkt_times) for fids, pkt_times, pkts in
                                                flows_tmp]
                        stat_data(np.asarray(self.flows_durations, dtype=float).reshape(-1, 1), name='flows_durations')

                        self.subflow_interval = self.params['subflow_interval']
                        if type(self.subflow_interval) == float and \
                                (self.subflow_interval > 0) and (self.subflow_interval < max(self.flows_durations)) \
                                and (self.subflow_interval != None):
                            print(f'+++subflow_interval: ', self.subflow_interval, ', q_flows_durations:',
                                  self.params['q_flow_dur'])
                        else:
                            print(
                                f'intervals: {np.quantile(self.flows_durations, q=[0.25, 0.5, 0.75, 0.9, 0.95])}, when q=[0.25, 0.5, 0.75, 0.9, 0.95]')
                            # self.q_flow_dur = 0.5 # come from PCAP() default is 0.5
                            self.q_flow_dur = self.params['q_flow_dur']
                            self.subflow_interval = np.quantile(self.flows_durations,
                                                                q=self.q_flow_dur)  # median  of flow_durations
                            print(f'---subflow_interval: ', self.subflow_interval,
                                  f', q_flow_dur: {self.q_flow_dur}')
                            self.params['subflow_interval'] = self.subflow_interval
                            # self.params['q_flow_dur'] = self.q_flow_dur
                    else:  # subflow not changes
                        raise ValueError('the first pcap must be normal.pcap, which is used to get interval.')
                flows_i = _load_pcap_to_subflows(pcap_file, num_pkt_thresh=self.num_pkt_thresh, max_pkts=-1,
                                                 interval=self.subflow_interval)  # all subflows
            # print("Number of FIDs in each subflow: {}".format([len(fids) for fids in list(zip(*flows))[0]]))
            else:
                flows_i, num_pkts = pcap2flows(pcap_file,
                                               self.num_pkt_thresh)  # get all flows which at least has more than 2 packets
            flows.extend(flows_i)
            print(f'num. of flows_i: {len(flows_i)}')
            if 'normal.pcap' in pcap_file and not 'abnormal.pcap' in pcap_file:
                labels_i = ['Normal'.upper()] * len(flows_i)
            elif 'anomaly.pcap' in pcap_file or 'abnormal.pcap' in pcap_file:
                labels_i = ['Anomaly'.upper()] * len(flows_i)
            else:
                labels_i = ['None'] * len(flows_i)
            labels.extend(labels_i)
        print(f'num. of flows: {len(flows)}')
        print(f'num. of labels: {Counter(labels)}')

        return flows, labels

    def plot_hist(self, data='', bins=50, title='', title_flg=True,
                  xlabel='duration of flow', ylabel='Counts', rescale_flg=False):

        # plt.subplot()  # plt.subplot(131)
        plt.subplots()  # plt.subplot(131)
        quant = 0.9
        data_thres = np.quantile(data, q=quant)
        data_thres = f'{data_thres:.4f}'
        print(f'data_thres: {data_thres} when quantile = {quant}')
        print(f'Counter(data)({len(data)}): {sorted(Counter(data).items(), key=lambda x: x[0], reverse=False)}')

        if rescale_flg:
            data = [value for value in data if value < 1]
            bins = 30

        plt.hist(data, bins=bins)  # arguments are passed to np.histogram

        hist, bin_edges = np.histogram(data, bins=bins)
        print(f'hist({len(hist)}):{hist},\nbin_edges({len(bin_edges)}):{bin_edges}')
        # max_idx = np.argmax(hist)
        # if max_idx - 1 >= 0:
        #     max_range = f'[{int(bin_edges[max_idx-1])}, {int(bin_edges[max_idx])}]'
        # else:
        #     max_range = f'[0, {int(bin_edges[max_idx])}]'
        # min_idx = np.argmin(hist)
        # if min_idx - 1 >= 0:
        #     min_range = f'[{int(bin_edges[min_idx-1])}, {int(bin_edges[min_idx])}]'
        # else:
        #     min_range = f'[0, {int(bin_edges[min_idx])}]'
        #
        # # title = f'srcIP:{srcIP},\nmax:{max(hist)} in {max_range},' \
        # #         f'\nmin:{min(hist)} in {min_range}'
        min_data = f'{min(data):.4f}'
        max_data = f'{max(data):.4f}'
        title += f'\nmin:{min_data}, max:{max_data}. {data_thres}, when q={quant}'
        if title_flg:
            plt.title(f"{title}")
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.subplots_adjust(bottom=0.25, top=0.75, left=0.005, right=0.99)
        plt.show()


def float2date(float_time):
    date_str = datetime.datetime.fromtimestamp(float_time).strftime('%Y-%m-%d %H:%M:%S')
    return date_str


def str2time(str_time):
    float_time = time.mktime(time.strptime(str_time, '%Y-%m-%d %H:%M:%S'))
    return float_time


def random_select_flows(flows, labels, experiment='ind', train_size=10000, test_size=1000,
                        random_state=42, pcap_file=''):
    """ obtain normal and anomaly flows and drop the rest.

    Parameters
    ----------
    flows
    labels
    experiment
    random_state

    Returns
    -------

    """
    # if experiment.upper() in ['INDV', 'MIX']:  # for indv and mix use all of data.
    #     return flows, labels

    cnt_normal = 0
    cnt_anomaly = 0
    others = []
    print(Counter(labels))
    for i, label_i in enumerate(labels):
        if label_i.upper() in ['NORMAL', 'BENIGN']:
            cnt_normal += 1
        elif label_i.upper() in ['BOT', 'ANOMALY', 'MALICIOUS']:
            cnt_anomaly += 1
        else:
            others.append(label_i)

    print(
        f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}, cnt_others: {len(others)}, Counter(others): '
        f'{Counter(others)}')
    get_all_flows_flg = True  # True: use all of flows samples, False: random select part of flow samples.
    # if 'DS50_MAWI_WIDE' in pcap_file or 'DS40_CTU_IoT' in pcap_file:
    #     CNT_ANOMALY = 400
    if 'DS60_UChi_IoT' in pcap_file or 'DS20_PU_SMTV' in pcap_file:
        CNT_ANOMALY = int(test_size // 2)
    elif 'DS10_UNB_IDS' in pcap_file:
        if cnt_normal > int(7 / 3 * cnt_anomaly):  # normal: anomaly = 7:3
            CNT_ANOMALY = cnt_anomaly
        else:
            CNT_ANOMALY = int(cnt_normal * 3 / 7)
    else:
        CNT_ANOMALY = int(test_size // 2)

    if cnt_anomaly < CNT_ANOMALY:
        part_anomaly_thres = cnt_anomaly
    else:
        part_anomaly_thres = CNT_ANOMALY
    if cnt_anomaly < 10:
        print(f'skip cnt_anomaly(={part_anomaly_thres}) < 10')
        print(f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}=> part_anomaly_thres: {part_anomaly_thres}')
        return -1

    part_normal_thres = train_size + part_anomaly_thres  # only random get 20000 normal samples.
    if cnt_normal > part_normal_thres or cnt_anomaly > part_anomaly_thres:
        get_all_flows_flg = False  # make all data have the same size
        # break # if has break here, it only print part of flows in cnt_normal

    print(f'before, len(flows): {len(flows)}, len(lables): {len(labels)}, get_all_flows_flg: {get_all_flows_flg}, '
          f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}')

    if not get_all_flows_flg:
        c = list(zip(flows, labels))
        flows_shuffle, labels_shuffle = zip(*shuffle(c, random_state=random_state))
        cnt_normal = 0
        cnt_anomaly = 0
        flows = []
        labels = []
        for i, (flows_i, label_i) in enumerate(zip(flows_shuffle, labels_shuffle)):
            if label_i.upper() in ['NORMAL', 'BENIGN']:
                cnt_normal += 1
                if cnt_normal <= part_normal_thres:
                    flows.append(flows_i)
                    labels.append(label_i)
            elif label_i.upper() in ['BOT', 'ANOMALY', 'MALICIOUS']:
                cnt_anomaly += 1
                if cnt_anomaly <= part_anomaly_thres:
                    flows.append(flows_i)
                    labels.append(label_i)

            if cnt_normal > part_normal_thres and cnt_anomaly > part_anomaly_thres:
                break
        else:
            pass
    print(f'after: len(flows): {len(flows)}, len(lables): {len(labels)}, get_all_flows_flg: {get_all_flows_flg}, '
          f'cnt_normal: {min(cnt_normal, part_normal_thres)}, cnt_anomaly: {min(cnt_anomaly, part_anomaly_thres)}')

    return flows, labels


def session_extractor(pkt):
    """Extract sessions from packets"""
    if IP in pkt and TCP in pkt:
        flow_type = 'TCP'
        fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
        return fid
    elif IP in pkt and UDP in pkt:
        flow_type = 'UDP'
        fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
        return fid

    return 'other'


def parse_tcp_flgs(tcp_flgs):
    # flags = {
    #     'F': 'FIN',
    #     'S': 'SYN',
    #     'R': 'RST',
    #     'P': 'PSH',
    #     'A': 'ACK',
    #     'U': 'URG',
    #     'E': 'ECE',
    #     'C': 'CWR',
    # }
    flgs = {
        'F': 0,
        'S': 0,
        'R': 0,
        'P': 0,
        'A': 0,
        'U': 0,
        'E': 0,
        'C': 0,
    }
    # flags = sorted(flags.items(), key=lambda x:x[0], reverse=True)
    # flags = OrderedDict(flags.items())
    # flg_lst = [0]*len(flags)
    # [flags[x] for x in p.sprintf('%TCP.flags%')]
    # ['SYN', 'ACK']
    for flg in tcp_flgs:
        if flg in flgs.keys():
            flgs[flg] += 1

    return list(flgs.values())


def get_header_features(pkts):
    features = []
    flgs_lst = np.zeros((8, 1))
    for i, pkt in enumerate(pkts):
        if pkt.payload.proto == 6:  # tcp
            flgs_lst += np.asarray(parse_tcp_flgs(pkt.payload.payload.flags)).reshape(-1, 1)  # parses tcp.flgs
        # elif pkt.payload.proto ==17: # udp
        #   pass
        features.append(pkt.payload.ttl)
    # features.append(pkt.payload.payload.dport)    # add dport will get 100% accuracy

    flgs_lst = list(flgs_lst.flatten())
    flgs_lst.extend(features)  # add ttl to the end of flgs.
    features = flgs_lst
    return features


def _flows_to_iats_sizes(flows, feat_set='iat', verbose=True):
    '''Get IATs features from flows

       Arguments:
         flows (list) = representation returned from read_pcap

       Returns:
         features (list) = [(fid, IATs)]
    '''
    # convert Unix timestamp arrival times into interpacket intervals
    # calculate IATs
    # the flows should be a deep copy of original flows. copy.deepcopy(flows)
    # flows = copy.deepcopy(flows_lst)      # may takes too much time
    if verbose:  # for verifying the code
        cnt = 3  # only show 3 flows
        cnt_1 = cnt
        flg = False
        for i, (fid, times, pkts) in enumerate(flows):  # flows is a list [(fid, times, pkts)]
            sizes = [len(pkt) for pkt in pkts]
            iats = np.diff(times)
            if (0 in iats) or (len(iats[iats == 0])):  # if two packets have the same timestamp?
                l = min(len(times), 10)
                # only print part of data
                print(f'i: {i}, 0 in np.diff(times): fid: {fid}, times (part of times to display): {times[:l]}, '
                      f'sizes: {sizes[:l]}, one reason is that retransmitted packets have the same time '
                      f'in wireshark, please check the pcap')
                cnt += 1
                if cnt > 3:
                    flg = True
            if sum(iats) == 0:  # flow's duration is 0.0. Is it possible?
                # One reason is that the flow only have two kinds of packets:
                # one is the sent packet, the rest of them  is the retransmitted packets which has the same time
                # to the sent packet in wireshark, please check
                print(f'i: {i}, sum(np.diff(times)) == 0:  fid: {fid}, times: {times}, sizes: {sizes}')
                cnt_1 += 1
                if cnt_1 > 3:
                    flg = True
            if flg:
                break

    if feat_set == 'iat':
        features = [(fid, np.asarray(list(np.diff(times)))) for (fid, times, pkts) in
                    flows]  # (fid, np.array())
    elif feat_set == 'size':
        features = [(fid, np.asarray([len(pkt) for pkt in pkts])) for (fid, times, pkts) in
                    flows]  # (fid, np.array())
    elif feat_set == 'iat_size':
        # features = [(fid, np.asarray(list(np.diff(times)) + [len(pkt) for pkt in pkts])) for (fid, times, pkts) in
        #             flows]  # (fid, np.array())
        features = []
        for (fid, times, pkts) in flows:
            feat = []
            feat_1 = list(np.diff(times))
            feat_2 = [len(pkt) for pkt in pkts]
            for i in range(len(times) - 1):
                feat.extend([feat_1[i], feat_2[i]])
            feat.append(feat_2[-1])
            features.append((fid, np.asarray(feat, dtype=float)))

    else:
        raise NotImplementedError(
            f'{feat_set} is not implemented, {os.path.relpath(_flows_to_iats_sizes.__code__.co_filename)}' \
            f'at line {_flows_to_iats_sizes.__code__.co_firstlineno}\'')

    return features


def _get_header_from_flows(flows):
    # convert Unix timestamp arrival times into interpacket intervals
    flows = [(fid, np.diff(times), pkts) for (fid, times, pkts) in flows]  # No need to use sizes[1:]
    features_header = []
    for fid, times, pkts in flows:  # fid, IAT, pkt_len
        features_header.append((fid, get_header_features(pkts)))  # (fid, np.array())

    return features_header


def _get_statistical_info(data):
    """

    Parameters
    ----------
    data: len(pkt)

    Returns
    -------
        a list includes mean, median, std, q1, q2, q3, min, and max.

    """
    q1, q2, q3 = np.quantile(data, q=[0.25, 0.5, 0.75])  # q should be [0,1] and q2 is np.median(data)
    return [np.mean(data), np.std(data), q1, q2, q3, np.min(data), np.max(data)]


def _flows_to_stats(flows):
    '''Converts flows to FFT features

       Arguments:
         flows (list) = representation returned from read_pcap


       Returns:
         features (list) = [(fid, (max, min, ... ))]
    '''
    # the flows should be a deep copy of original flows. copy.deepcopy(flows)
    # flows = copy.deepcopy(flows_lst)

    # convert Unix timestamp arrival times into interpacket intervals
    flows = [(fid, np.diff(times), pkts) for (fid, times, pkts) in flows]  # No need to use sizes[1:]
    # len(np.diff(times)) + 1  == len(sizes)
    features = []
    features_header = []
    for fid, times, pkts in flows:  # fid, IAT, pkt_len
        sizes = [len(pkt) for pkt in pkts]
        sub_duration = sum(times)  # Note: times here actually is the results of np.diff()
        num_pkts = len(sizes)  # number of packets in the flow
        num_bytes = sum(sizes)  # all bytes in sub_duration  sum(len(pkt))
        if sub_duration == 0:
            pkts_rate = 0.0
            bytes_rate = 0.0
        else:
            pkts_rate = num_pkts / sub_duration  # it will be very larger due to the very small sub_duration
            bytes_rate = num_bytes / sub_duration
        base_feature = [sub_duration, pkts_rate, bytes_rate] + _get_statistical_info(sizes)

        features.append(
            (fid, np.asarray([np.float64(v) for v in base_feature], dtype=np.float64)))  # (fid, np.array())

    return features


def handle_large_time_diff(start_time, end_time, interval=0.1, max_num=10000):
    """

    :param start_time:
    :param end_time:
    :param interval:
    :param max_num: the maximum number of 0 inserted to the features
    :return:
    """
    if start_time >= end_time:
        raise ValueError('start_time >= end_time')

    num_intervals = int((end_time - start_time) // interval)
    # print(f'num_intervals: {num_intervals}')
    if num_intervals > max_num:
        # print(
        #     f'num_intervals with 0: {num_intervals} = (end_time({end_time}) -
        #     start_time({start_time}))/(sampling_rate: {interval})'
        #     f', only keep: {max_num}')
        num_intervals = max_num
    features_lst = [0] * num_intervals

    start_time = start_time + num_intervals * interval

    return features_lst, start_time


def sampling_packets(flow, sampling_type='rate', sampling=5, sampling_feature='samp_num', random_state=42):
    """

    :param flow:
    :param sampling_type:
    :param sampling:
    :return:
    """
    # the flows should be a deep copy of original flows. copy.deepcopy(flow)

    fid, times, sizes = flow
    sampling_data = []

    if sampling_type == 'rate':  # sampling_rate within flows.

        # The length in time of this small window is what we’re calling sampling rate.
        # features obtained on sampling_rate = 0.1 means that:
        #  1) split each flow into small windows, each window has 0.1 duration (the length in time of each small window)
        #  2) obtain the number of packets in each window (0.1s).
        #  3) all the number of packets in each window make up of the features.

        if sampling_feature in ['samp_num', 'samp_size']:
            features = []
            samp_sub = 0
            # print(f'len(times): {len(times)}, duration: {max(times)-min(times)}, sampling: {sampling},
            # num_features: {int(np.round((max(times)-min(times))/sampling))}')
            for i in range(len(times)):  # times: the arrival time of each packet
                if i == 0:
                    current = times[0]
                    if sampling_feature == 'samp_num':
                        samp_sub = 1
                    elif sampling_feature == 'samp_size':
                        samp_sub = sizes[0]
                    continue
                if times[i] - current <= sampling:  # interval
                    if sampling_feature == 'samp_num':
                        samp_sub += 1
                    elif sampling_feature == 'samp_size':
                        samp_sub += sizes[i]
                    else:
                        print(f'{sampling_feature} is not implemented yet')
                else:  # if times[i]-current > sampling:    # interval
                    current += sampling
                    features.append(samp_sub)
                    # the time diff between times[i] and times[i-1] will be larger than mutli-samplings
                    # for example, times[i]=10.0s, times[i-1]=2.0s, sampling=0.1,
                    # for this case, we should insert int((10.0-2.0)//0.1) * [0]
                    num_intervals = int(np.floor((times[i] - current) // sampling))
                    if num_intervals > 0:
                        num_intervals = min(num_intervals, 500)
                        features.extend([0] * num_intervals)
                        current += num_intervals * sampling
                    # if current + sampling <= times[i]:  # move current to the nearest position to time[i]
                    #     feat_lst_tmp, current = handle_large_time_diff(start_time=current, end_time=times[i],
                    #                                                    interval=sampling)
                    # features.extend(feat_lst_tmp)
                    if len(features) > 500:  # avoid num_features too large to excess the memory.
                        return fid, features[:500]

                    # samp_sub = 1  # includes the time[i] as a new time interval
                    if sampling_feature == 'samp_num':
                        samp_sub = 1
                    elif sampling_feature == 'samp_size':
                        samp_sub = sizes[i]

            if samp_sub > 0:  # handle the last sub period in the flow.
                features.append(samp_sub)

            return fid, features
        else:
            raise ValueError(f'sampling_feature: {sampling_feature} is not implemented.')
    else:
        raise ValueError(f'sample_type: {sampling_type} is not implemented.')


def _flows_to_samps(flows, sampling_type='rate', sampling=None,
                    sampling_feature='samp_num',
                    verbose=True):
    """ sampling packets in flows
    Parameters
    ----------
    flows
    sampling_type
    sampling
    sampling_feature
    verbose

    Returns
    -------

    """
    # the flows should be a deep copy of original flows. copy.deepcopy(flows)
    # flows = copy.deepcopy(flows_lst)

    # samp_flows = []
    features = []
    features_header = []
    for fid, times, pkts in flows:
        sizes = [len(pkt) for pkt in pkts]
        if sampling_feature == 'samp_num_size':
            samp_features = []
            samp_fid_1, samp_features_1 = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
                                                           sampling=sampling, sampling_feature='samp_num')

            samp_fid_2, samp_features_2 = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
                                                           sampling=sampling, sampling_feature='samp_size')
            for i in range(len(samp_features_1)):
                if len(samp_features) > 500:
                    break
                samp_features.extend([samp_features_1[i], samp_features_2[i]])
            samp_fid = samp_fid_1
        else:
            samp_fid, samp_features = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
                                                       sampling=sampling, sampling_feature=sampling_feature)

        features.append((samp_fid, samp_features))  # (fid, np.array())

    # if header:
    #     head_len = int(np.quantile([len(head) for (fid, head) in features_header], q=q_iat))
    #     for i, (fid_head, fid_feat) in enumerate(list(zip(features_header, features))):
    #         fid, head = fid_head
    #         fid, feat = fid_feat
    #         if len(head) > head_len:
    #             head = head[:head_len]
    #         else:
    #             head += [0] * (head_len - len(head))
    #         features[i] = (fid, np.asarray(head + list(feat)))

    if verbose:  # for debug
        show_len = 10  # only show the first 20 difference
        samp_lens = np.asarray([len(samp_features) for (fid, samp_features) in features])[:show_len]
        raw_lens = np.asarray([max(times) - min(times) for (fid, times, sizes) in flows])[:show_len]
        print(f'(flow duration, num_windows), when sampling_rate({sampling})):\n{list(zip(raw_lens, samp_lens))}')

    return features


def _load_labels_and_label_flows_by_data(fids_labels, features='', label_file_type='CTU-IoT-23'):
    """ label features by fids_labels

    Parameters
    ----------
    fids_labels
    features
    label_file_type

    Returns
    -------

    """

    labels = []
    for i, (fid, feat) in enumerate(features):
        flg = False
        for j, (fid_j, label_j) in enumerate(fids_labels):
            if fid == fid_j:
                flg = True
                labels.append(label_j)
                break

        if not flg:
            labels.append('None')
    print(f'len(labels): {len(labels)}')

    return labels


@memory_usuage
def _load_labels_and_label_flows(label_file='csv_file', features='', label_file_type='CTU-IoT-23'):
    '''Load binary labels from CICIDS format CSV

       Arguments:
         label_file: csv_file (string): path to CSV file

       Returns:
         labels[(5-tuple flow ID)] -> label (string) (e.g. "BENIGN")
    '''
    if label_file_type == 'CTU-IoT-23':
        raise NotImplementedError
    else:
        NORMAL_LABELS = [v.upper() for v in ['benign', 'normal']]
        ANOMALY_LABELS = [v.upper() for v in ['ANOMALY', 'Malicious', 'FTP-PATATOR', 'SSH-PATATOR',
                                              'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
                                              'Heartbleed',
                                              'Web Attack – Brute Force', 'Web Attack – XSS',
                                              'Web Attack – Sql Injection', 'Infiltration',
                                              'Bot', 'PortScan', 'DDoS']]

        # load CSV with pandas
        csv = pd.read_csv(label_file)

        labels = {}
        cnt_anomaly = 0
        cnt_nomral = 0
        others = 0
        cnt_conflicts = 0
        for i, r in enumerate(csv.index):
            if i % 10000 == 0:
                print("Label CSV row {}".format(i))
            row = csv.loc[r]
            # parse 5-tuple flow ID
            if 'LABEL' in row[" Label"].upper():
                continue
            fid = (str(row[" Source IP"]), str(row[" Destination IP"]), int(row[" Source Port"]),
                   int(row[" Destination Port"]), int(row[" Protocol"]))
            # ensure all 5-tuple flows have same label
            label_i = row[" Label"].upper()
            if label_i in ANOMALY_LABELS:
                label_i = 'ANOMALY'
                cnt_anomaly += 1
            elif label_i in NORMAL_LABELS:
                label_i = 'NORMAL'
                cnt_nomral += 1
            else:
                others += 1
            if fid in labels.keys():  # a session has different labels, is it possible?
                if labels[fid] != label_i:
                    cnt_conflicts += 1
                    print('labels[fid] != label_i', labels[fid], label_i)
                # assert (labels[fid] == row[" Label"]), f'{i}, {fid}, {labels[fid]}'  # one fid has different labels
                # if labels[fid] != row[" Label"]:
                #     print(labels[fid], row[" Label"])
                # else:
                #     # print(labels[fid], row[" Label"])
                #     pass
            # set label of flow ID
            labels[fid] = label_i
        print(f'label_csv includes: cnt_normal: {cnt_nomral}, cnt_anomaly: {cnt_anomaly}, others: {others}, '
              f'Unique labels: '
              f'Counter(labels.values()),{Counter(labels.values())}, cnt_conflicts {cnt_conflicts}')
        # tmp_lbs = [ k for k in labels.keys()]
        # print(f'anomaly labels: {Counter(tmp_lbs)}')

        # obtain the labels of the corresponding features
        new_labels = []
        not_existed_fids = []
        new_fids = []
        for i, (fid, feat) in enumerate(features):
            if i == 0:
                print(fid, list(labels.keys())[0])
            if fid in labels.keys():
                new_labels.append(labels[fid])
                new_fids.append(fid)
            else:
                not_existed_fids.append(fid)
                new_fids.append('None')
                new_labels.append('None')  # the fid does not exist in labels.csv

    print(f'{len(not_existed_fids)} (unique fids: {len(set(not_existed_fids))}) flows do not exist in {label_file},'
          f'Counter(not_existed_fids)[:10]{list(Counter(not_existed_fids))[:10]}')
    print(f'len(new_labels): {len(new_labels)}, unique labels of new_labels: {Counter(new_labels)}')
    return new_fids, new_labels


def merge_files_to_one(file_lst=[], mixed_file='', verbose=True):
    if verbose:
        funcparams_dict = {'file_lst': file_lst, 'mixed_file': mixed_file}
        pprint(OrderedDict(funcparams_dict), name=merge_files_to_one.__name__)
    check_n_generate_path(file_path=mixed_file, overwrite=True)

    with open(mixed_file, 'wb') as out_hdl:
        for idx, file_path in enumerate(file_lst):
            print(f'*index: {idx}, file_path: {file_path}')
            with open(file_path, 'rb') as in_hdl:
                fids, features, labels = pickle.load(in_hdl)
                pickle.dump((fids, features, labels), out_hdl)

    print(f'mixed_file: {mixed_file}')

    return mixed_file


def merge_pcaps(pcap_file_lst=[], mrg_pcap_path=''):
    if os.path.exists(mrg_pcap_path):
        os.remove(mrg_pcap_path)
    if not os.path.exists(os.path.dirname(mrg_pcap_path)):
        os.makedirs(os.path.dirname(mrg_pcap_path))
    cmd = f"mergecap -w {mrg_pcap_path} " + ' '.join(pcap_file_lst)
    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}')
        return -1


def merge_labels(label_file_lst=[], mrg_label_path=''):
    print(label_file_lst, mrg_label_path)
    if os.path.exists(mrg_label_path):
        os.remove(mrg_label_path)

    if not os.path.exists(os.path.dirname(mrg_label_path)):
        os.makedirs(os.path.dirname(mrg_label_path))
    # # combine all label files in the list
    # # combined_csv = pd.concat([pd.read_csv(f, header=None, usecols=[3,6]) for f in label_file_lst])
    # result_lst = []
    # for i, f in enumerate(label_file_lst):
    #     if i == 0:
    #         result_lst.append(pd.read_csv(f))
    #     else:
    #         result_lst.append(pd.read_csv(f, skiprows=0))
    # combined_csv = pd.concat(result_lst)
    # # export to csv
    # print(f'mrg_label_path: {mrg_label_path}')
    # combined_csv.to_csv(mrg_label_path, index=False, encoding='utf-8-sig')

    with open(mrg_label_path, 'w') as out_f:
        header = True
        for i, label_file in enumerate(label_file_lst):
            with open(label_file, 'r') as in_f:
                line = in_f.readline()
                while line:
                    if line.strip().startswith('Flow ID') and header:
                        if header:
                            header = False
                            print(line)
                            out_f.write(line.strip('\n') + '\n')
                        else:
                            pass
                        line = in_f.readline()
                        continue
                    if line.strip() == '':
                        line = in_f.readline()
                        continue
                    out_f.write(line.strip('\n') + '\n')
                    line = in_f.readline()

    return mrg_label_path


def merge_csvs(label_files, output_file=''):
    with open(output_file, 'w') as out_f:
        header = True
        for i, label_file in enumerate(label_files):
            with open(label_file, 'r') as in_f:
                line = in_f.readline()
                while line:
                    if line.strip().startswith('Flow'):
                        if header:
                            header = False
                            print(line)
                            out_f.write(line.strip('\n') + '\n')
                        else:
                            pass
                        line = in_f.readline()
                        continue
                    if line.strip() == '':
                        line = in_f.readline()
                        continue
                    out_f.write(line.strip('\n') + '\n')
                    line = in_f.readline()

    return output_file


#
# def generate_label(pcap_file, label='Normal', num_pkt_thresh=2):
#     # pcap_file = '-input_data/smart-tv-roku-data/multi-srcIPs/pcaps/merged_normal.pcap'  # for testing
#     flows = pcap2flows(pcap_file, num_pkt_thresh)  # get all flows which at least has more than 2 packets
#     # fids = [[fid, label] for fid, _, _ in flows]
#     label_file = os.path.splitext(pcap_file)[0] + '.csv'
#     if os.path.exists(label_file):
#         os.remove(label_file)
#
#     # pd.DataFrame(fids).to_csv(label_file)
#     with open(label_file, 'w') as out_hdl:
#         header = [" Source IP", " Destination IP", " Source Port", " Destination Port", " Protocol", " Label"]
#         out_hdl.write(",".join(header) + '\n')
#         for i, (fid, _, _) in enumerate(flows):
#             line_lst = [str(v) for v in fid]
#             line_lst.append(str(label))
#             out_hdl.write(','.join(line_lst) + '\n')
#
#     return label_file


def conn_log_2_csv(input_file, output_file=''):
    print(f'input_file: {input_file}')
    if output_file == '':
        output_file = input_file + '.csv'
    with open(output_file, 'w') as f_out:
        with open(input_file, 'r') as f_in:
            line = f_in.readline()
            while line:
                if line.strip().startswith('#'):
                    if line.strip().startswith('#fields'):
                        # id.orig_h id.orig_p   id.resp_h   id.resp_p
                        line = line.replace('id.orig_h', 'id.orig_h(srcIP)').replace('id.orig_p', 'id.orig_p(sport)') \
                            .replace('id.resp_h', 'id.resp_h(dstIP)').replace('id.resp_p', 'id.resp_p(dport)')
                        arr = [v.strip() for v in line.split()]
                        f_out.write(','.join(arr[1:]) + '\n')
                    line = f_in.readline()
                    continue
                else:
                    arr = [v.strip() for v in line.split()]
                    f_out.write(",".join(arr) + '\n')
                line = f_in.readline()

    print(f'output_file: {output_file}')
    return output_file


def process_CIC_IDS_2017(label_file, time_range=['start', 'end'], output_file='_reduced.txt'):
    """ timezone: ADT in CICIDS_2017 label.csv

    Parameters
    ----------
    label_file
    time_range
    output_file

    Returns
    -------

    """
    with open(output_file, 'w') as out_f:
        start = 0
        i = 0
        start_flg = True
        end = 0
        max_sec = -1
        min_sec = -1
        with open(label_file, 'r') as in_f:
            line = in_f.readline()
            flg = False
            while line:
                if line.startswith("Flow"):
                    line = in_f.readline()
                    continue
                arr = line.split(',')
                # time
                # print(arr[6])
                time_str = datetime.strptime(arr[6], "%d/%m/%Y %H:%M")
                time_str = convert_datetime_timezone(str(time_str), tz1='Canada/Atlantic', tz2='UTC')
                ts = time_string_to_seconds(str(time_str), '%Y-%m-%d %H:%M:%S')
                if start_flg:
                    print(i, ts, start)
                    start = ts
                    min_sec = start
                    start_flg = False
                else:
                    if ts > end:
                        end = ts
                    if ts < min_sec:
                        min_sec = ts
                    if ts > max_sec:
                        max_sec = ts
                if ts > time_range[0] and ts < time_range[1]:
                    out_f.write(line.strip('\n') + '\n')
                # if ts > time_range[1]:
                #     break

                line = in_f.readline()
                i += 1
        print(start, end, time_range, i, min_sec, max_sec)

    return output_file


def extract_packets(ip_file, pcap_file):
    with open(ip_file, 'r') as in_f:
        line = in_f.readline()
        i = 1
        while line:
            arr = line.split(':')
            ip = arr[0]
            num_packets = int(arr[1])
            if len(ip.split('.')) == 4 and num_packets > 100:
                output_file = pcap_file + f'-IP_{ip}.pcap'
                cmd = f"tshark -r {pcap_file} -w {output_file} ip.addr=={ip}"
                print(f'{cmd}')
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
                except Exception as e:
                    print(f'{e}, {result}')
                    continue
            else:
                print(f"{i}, line: {line}")
                break
            line = in_f.readline()
            i += 1


def _extract_packets(pcap_file, ip_cmd='', output_file=''):
    out_dir = pth.dirname(output_file)
    if not pth.exists(out_dir):
        os.makedirs(out_dir)

    cmd = f"tshark -r {pcap_file} -w {output_file} {ip_cmd}"
    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}')


def extract_packets_from_pcap(pcap_file, srcIPs=[], dstIPs=[], IPs=[], type='individual', out_dir=''):
    if type == 'individual':
        for ip in srcIPs:
            if not pth.exists(out_dir):
                os.makedirs(out_dir)
            output_file = pth.join(out_dir, f'srcIP_{ip}.pcap')
            ip_cmd = f'ip.src=={ip}'
            _extract_packets(pcap_file, ip_cmd, output_file)

        for ip in dstIPs:
            if not pth.exists(out_dir):
                os.makedirs(out_dir)
            output_file = pth.join(out_dir, f'dstP_{ip}.pcap')
            ip_cmd = f'ip.dst=={ip}'
            _extract_packets(pcap_file, ip_cmd, output_file)

        for ip in IPs:
            if not pth.exists(out_dir):
                os.makedirs(out_dir)
            output_file = pth.join(out_dir, f'IP_{ip}.pcap')
            ip_cmd = f'ip.addr=={ip}'
            _extract_packets(pcap_file, ip_cmd, output_file)
    else:
        _name = "-".join(srcIPs)
        if not pth.exists(out_dir):
            os.makedirs(out_dir)
        output_file = pth.join(out_dir, f'-{_name}.pcap')
        ip_cmd = 'ip.src==' + " or ip.src==".join(srcIPs)
        _extract_packets(pcap_file, ip_cmd, output_file)

        _name = "-".join(dstIPs)
        if not pth.exists(out_dir):
            os.makedirs(out_dir)
        output_file = pth.join(out_dir, f'-{_name}.pcap')
        ip_cmd = 'ip.dst==' + " or ip.dst==".join(dstIPs)
        _extract_packets(pcap_file, ip_cmd, output_file)

        _name = "-".join(IPs)
        if not pth.exists(out_dir):
            os.makedirs(out_dir)
        output_file = pth.join(out_dir, f'-{_name}.pcap')
        ip_cmd = 'ip.addr==' + " or ip.addr==".join(IPs)
        _extract_packets(pcap_file, ip_cmd, output_file)

    return output_file


def get_time(time_str, time_format="%Y-%m-%d %H:%M:%S"):
    sec = 0
    try:  # 6/7/2017 10:18
        date_time = datetime.strptime(time_str, time_format)
        sec = (date_time - datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, microsecond=0,
                                    tzinfo=None)).total_seconds()
    except Exception as e:
        time_formats = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
        for time_format in time_formats:
            try:  # 6/7/2017 10:18
                date_time = datetime.strptime(time_str, time_format)
                sec = (date_time - datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, microsecond=0,
                                            tzinfo=None)).total_seconds()
                break
            except Exception as e:
                continue
    return float(str(sec))


def pcap2flows(pth_pcap, num_pkt_thresh=2, verbose=True):
    '''Reads pcap and divides packets into 5-tuple flows (arrival times and sizes)

           Arguments:
             pcap_file (string) = path to pcap file
             num_pkt_thresh (int) = discards flows with fewer packets than max(2, thresh)

           Returns:
             flows (list) = [(fid, arrival times list, packet sizes list)]
        '''

    print(f'pcap_file: {pth_pcap}')
    sessions = OrderedDict()  # key order of fid by time
    num_pkts = 0
    "filter pcap_file only contains the special srcIP "
    try:
        # sessions= rdpcap(pcap_file).sessions()
        # res = PcapReader(pcap_file).read_all(count=-1)
        # from scapy import plist
        # sessions = plist.PacketList(res, name=os.path.basename(pcap_file)).sessions()
        for i, pkt in enumerate(PcapReader(pth_pcap)):  # iteratively get packet from the pcap
            if i % 10000 == 0:
                print(f'i_pkt: {i}')
            sess_key = session_extractor(pkt)  # this function treats bidirection as two sessions.
            # if ('TCP' in sess_key) or ('UDP' in sess_key) or (6 in sess_key) or (17 in sess_key):
            if (TCP in pkt) or (UDP in pkt):
                if sess_key not in sessions.keys():
                    sessions[sess_key] = [pkt]
                else:
                    sessions[sess_key].append(pkt)

    except Exception as e:
        print('Error', e)

    print(f'len(sessions) {len(sessions.keys())}')

    def get_frame_time(pkt):
        return float(pkt.time)

    # # in order to reduce the size of sessions and sort the pkt by time for the latter part.
    # new_sessions = OrderedDict()
    # for i, (key, sess) in enumerate(sessions.items()):
    #     if len(sess) >= max(2, num_pkt_thresh):
    #         new_sessions[key] = sorted(sess, key=get_frame_time, reverse=False)
    # here it will spend too much time, however, it is neccessary to do that.
    #     # sessions[key] = sorted(sess, key=lambda pkt: float(pkt.time), reverse=False)
    #
    # sessions = new_sessions

    flows = []  # store all the flows

    TCP_TIMEOUT = 600  # 600seconds, 10 mins
    UDP_TIMEOUT = 600  # 10mins.

    remainder_cnt = 0
    new_cnt = 0  # a flow is not split by an interval
    num_pkts = 0
    for i, (key, sess) in enumerate(sessions.items()):  # split sessions by TIMEOUT:
        if len(sess) >= max(2, num_pkt_thresh):
            sess = sorted(sess, key=get_frame_time,
                          reverse=False)  # here it will spend too much time, however, it is neccessary to do that.
            # sessions[key] = sorted(sess, key=lambda pkt: float(pkt.time), reverse=False)
        else:
            continue

        num_pkts += len(sess)
        # print(f'session_i: {i}')
        flow_i = []
        flow_type = None
        subflow = []
        new_flow = 0
        for j, pkt in enumerate(sess):
            if TCP not in pkt and UDP not in pkt:
                break
            if j == 0:
                subflow = [(float(pkt.time), pkt)]
                split_flow = False  # if a flow is not split with interval, label it as False, otherwise, True
                continue
            # handle TCP packets
            if IP in pkt and TCP in pkt:
                flow_type = 'TCP'
                fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
                if float(pkt.time) - subflow[-1][0] > TCP_TIMEOUT:
                    # timeout between the previous pkt and the current one, which is the idle time, not the
                    # subflow duration
                    flow_i.append((fid, subflow))
                    subflow = [(float(pkt.time), pkt)]
                    split_flow = True
                else:
                    subflow.append((float(pkt.time), pkt))

            # handle UDP packets
            elif IP in pkt and UDP in pkt:
                # parse 5-tuple flow ID
                fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
                flow_type = 'UDP'
                if float(pkt.time) - subflow[-1][0] > UDP_TIMEOUT:
                    flow_i.append((fid, subflow))
                    subflow = [(float(pkt.time), pkt)]
                    split_flow = True
                else:
                    subflow.append((float(pkt.time), pkt))

        if (flow_type in ['TCP', 'UDP']):
            flow_i.append((fid, subflow))

        flows.extend(flow_i)
    n_lq_2 = len([len(v) for s, v in sessions.items() if len(v) < 2])
    n_gq_2 = len([len(v) for s, v in sessions.items() if len(v) >= 2])
    print(f'all sessions: {len(sessions.keys())}, num of sessions < 2 pkts: {n_lq_2}, >=2: {n_gq_2}')
    print(f'all flows: {len(flows)} after discarding the flow that has less than 2 pkts, and using timeout '
          f'split (the times UDP packets vary)')
    # sort all flows by packet arrival time, each flow must have at least two packets
    # flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in flows if
    #          len(times_pkts) >= max(2, num_pkt_thresh)]
    flows = [(fid, *list(zip(*times_pkts))) for fid, times_pkts in flows if
             len(times_pkts) >= max(2, num_pkt_thresh)]
    print(f'the final stats: len(flows): {len(flows)}, each of them has more than 2 pkts (using timeout (10 mins) '
          f'split, some flows with 2 packets will be split as two flows, i.e., each of them just has one packet, '
          f'so we drop them).')

    flows_durations = [max(pkts_times) - min(pkts_times) for fid, pkts_times, pkts in flows]
    stat_data(np.asarray(flows_durations, dtype=float).reshape(-1, 1), name='flows_durations')

    return flows, num_pkts


def _load_pcap_to_subflows(pcap_file, num_pkt_thresh=2, interval=0.01, max_pkts=-1, verbose=True):
    '''Reads pcap and divides packets into 5-tuple flows (arrival times and sizes)

       Arguments:
         pcap_file (string) = path to pcap file
         num_pkt_thresh (int) = discards flows with fewer packets than max(2, thresh)

       Returns:
         flows (list) = [(fid, arrival times list, packet sizes list)]
    '''

    full_flows, num_pkts = pcap2flows(pcap_file, num_pkt_thresh=2, verbose=True)
    remainder_cnt = 0
    new_cnt = 0  # a flow is not split by an intervals
    flows = []  # store the subflows
    step_flows = []
    tmp_arr2 = []
    tmp_arr1 = []
    print(f'interval: {interval}')
    if ('anomaly' in pcap_file or 'abnormal' in pcap_file) and 'UCHI/IOT_2019' in pcap_file:
        if 'scam' in pcap_file or 'ghome' in pcap_file or 'sfirg' in pcap_file or 'bstch' in pcap_file:
            print('normal file does not need split with different steps, only anomaly file needs.')
    for i, (fid, times, pkts) in enumerate(full_flows):
        if i % 1000 == 0:
            print(f'session_i: {i}, len(pkts): {len(pkts)}')

        flow_type = None
        new_flow = 0
        dur = max(times) - min(times)
        if dur >= 2 * interval:
            tmp_arr2.append(max(times) - min(times))  # 10% flows exceeds the interals

        if dur >= 1 * interval:
            tmp_arr1.append(max(times) - min(times))
        step = 0
        # 'step' for 'normal data' always equals 0. If dataset needs to be agumented, then slide window with step
        while step < len(pkts):
            # print(f'i: {i}, step:{step}, len(pkts[{step}:]): {len(pkts[step:])}')
            dur_tmp = max(times[step:]) - min(times[step:])
            if dur_tmp <= interval:
                if step == 0:
                    subflow = [(float(pkt.time), pkt) for pkt in pkts[step:]]
                    step_flows.append((fid, subflow))
                    flows.append((fid, subflow))
                break  # break while loop
            flow_i = []
            subflow = []
            for j, pkt in enumerate(pkts[step:]):
                if TCP not in pkt and UDP not in pkt:
                    break
                if j == 0:
                    flow_start_time = float(pkt.time)
                    subflow = [(float(pkt.time), pkt)]
                    split_flow = False  # if a flow is not split with interval, label it as False, otherwise, True
                    continue
                # handle TCP packets
                if IP in pkt and TCP in pkt:
                    flow_type = 'TCP'
                    fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
                    if float(pkt.time) - flow_start_time > interval:
                        flow_i.append((fid, subflow))
                        flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
                        subflow = [(float(pkt.time), pkt)]
                        split_flow = True
                    else:
                        subflow.append((float(pkt.time), pkt))

                # handle UDP packets
                elif IP in pkt and UDP in pkt:
                    # parse 5-tuple flow ID
                    fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
                    flow_type = 'UDP'
                    if float(pkt.time) - flow_start_time > interval:
                        flow_i.append((fid, subflow))
                        flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
                        subflow = [(float(pkt.time), pkt)]
                        split_flow = True
                    else:
                        subflow.append((float(pkt.time), pkt))

            # drop the last one which interval is less than interval
            if step == 0:  # full flows
                flows.extend(flow_i)

            step_flows.extend(flow_i)  # for IoT dataset, we use step to get more flows.
            if ('anomaly' in pcap_file or 'abnormal' in pcap_file) and 'UCHI/IOT_2019' in pcap_file:
                if 'ghome' in pcap_file or 'sfirg' in pcap_file or 'bstch' in pcap_file:
                    step += 10
                elif 'scam' in pcap_file:  # when direction = 'src'
                    step += 5  # 'agument' anomaly files in DS60, scam has to less pakects in each flows, so step = 5
                else:
                    break
            else:
                break
    print(f'tmp_arr2: {len(tmp_arr2)},tmp_arr1: {len(tmp_arr1)}, all_flows: {len(full_flows)}, subflows: {len(flows)}'
          f', step_flows: {len(step_flows)}, pcap_file: {pcap_file}')
    # print(f'all subflows: {len(flows) + remainder_cnt}, new_flows ({new_cnt}) which durations are less than
    # the interval, old_flows (discarded): {remainder_cnt}')

    # sort all flows by packet arrival time, each flow must have at least two packets
    flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in flows if
             len(times_pkts) >= max(2, num_pkt_thresh)]
    print(f'the final subflows: len(flows): {len(flows)}, each of them has more than 2 pkts.')

    # sort all flows by packet arrival time, each flow must have at least two packets
    step_flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in step_flows if
                  len(times_pkts) >= max(2, num_pkt_thresh)]
    print(f'the final step_flows: len(step_flows): {len(step_flows)}, each of them has more than 2 pkts.')
    if 'anomaly' in pcap_file or 'abnormal' in pcap_file:
        return step_flows

    return flows


#
# def flows2subflows(full_flows, interval=10, num_pkt_thresh=2, data_name='', abnormal=False):
#     remainder_cnt = 0
#     new_cnt = 0  # a flow is not split by an intervals
#     flows = []  # store the subflows
#     step_flows = []
#     tmp_arr2 = []
#     tmp_arr1 = []
#     print(f'interval: {interval}')
#     print('normal file does not need split with different steps, only anomaly file needs.')
#     for i, (fid, times, pkts) in enumerate(full_flows):
#         if i % 1000 == 0:
#             print(f'session_i: {i}, len(pkts): {len(pkts)}')
#
#         flow_type = None
#         new_flow = 0
#         dur = max(times) - min(times)
#         if dur >= 2 * interval:
#             tmp_arr2.append(max(times) - min(times))  # 10% flows exceeds the interals
#
#         if dur >= 1 * interval:
#             tmp_arr1.append(max(times) - min(times))
#         step = 0  # 'step' for 'normal data' always equals 0. If dataset needs to be agumented, then slide window with step
#         while step < len(pkts):
#             # print(f'i: {i}, step:{step}, len(pkts[{step}:]): {len(pkts[step:])}')
#             dur_tmp = max(times[step:]) - min(times[step:])
#             if dur_tmp <= interval:
#                 if step == 0:
#                     subflow = [(float(pkt.time), pkt) for pkt in pkts[step:]]
#                     step_flows.append((fid, subflow))
#                     flows.append((fid, subflow))
#                 break  # break while loop
#             flow_i = []
#             subflow = []
#             for j, pkt in enumerate(pkts[step:]):
#                 if TCP not in pkt and UDP not in pkt:
#                     break
#                 if j == 0:
#                     flow_start_time = float(pkt.time)
#                     subflow = [(float(pkt.time), pkt)]
#                     split_flow = False  # if a flow is not split with interval, label it as False, otherwise, True
#                     continue
#                 # handle TCP packets
#                 if IP in pkt and TCP in pkt:
#                     flow_type = 'TCP'
#                     fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
#                     if float(pkt.time) - flow_start_time > interval:
#                         flow_i.append((fid, subflow))
#                         flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
#                         subflow = [(float(pkt.time), pkt)]
#                         split_flow = True
#                     else:
#                         subflow.append((float(pkt.time), pkt))
#
#                 # handle UDP packets
#                 elif IP in pkt and UDP in pkt:
#                     # parse 5-tuple flow ID
#                     fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
#                     flow_type = 'UDP'
#                     if float(pkt.time) - flow_start_time > interval:
#                         flow_i.append((fid, subflow))
#                         flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
#                         subflow = [(float(pkt.time), pkt)]
#                         split_flow = True
#                     else:
#                         subflow.append((float(pkt.time), pkt))
#
#             if (split_flow == False) and (flow_type in ['TCP', 'UDP']):
#                 new_cnt += 1
#                 flow_i.append((fid, subflow))
#             else:
#                 # drop the last subflow after splitting a flow
#                 remainder_cnt += 1
#                 # flow_i.append((fid, subflow)) # don't include the remainder
#                 # print(i, new_flow, subflow)
#
#             # drop the last one which interval is less than interval
#             if step == 0:
#                 flows.extend(flow_i)
#
#             step_flows.extend(flow_i)
#             if 'DS60_UChi_IoT' in data_name and abnormal:  # only augment abnormal flows
#                 step += 5  # 'agument' anomaly files in DS60
#             else:
#                 break
#
#             # if ('anomaly' in pcap_file or 'abnormal' in pcap_file) and 'UChi/IOT_2019' in pcap_file:
#             #     if 'scam' in pcap_file or 'ghome' in pcap_file or 'sfirg' in pcap_file or 'bstch' in pcap_file:
#             #         step += 5  # 'agument' anomaly files in DS60
#             # else:
#             #     break
#
#
#     print(
#         f'tmp_arr2: {len(tmp_arr2)},tmp_arr1: {len(tmp_arr1)}, all_flows: {len(full_flows)}, subflows:
#         {len(flows)}, step_flows: {len(step_flows)}, {data_name}, remain_subflow: {len(subflow)}')
#
#     # sort all flows by packet arrival time, each flow must have at least two packets
#     flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in flows if
#              len(times_pkts) >= max(2, num_pkt_thresh)]
#     print(f'the final subflows: len(flows): {len(flows)}, each of them has more than 2 pkts.')
#
#     # sort all flows by packet arrival time, each flow must have at least two packets
#     step_flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in step_flows if
#                   len(times_pkts) >= max(2, num_pkt_thresh)]
#     print(f'the final step_flows: len(step_flows): {len(step_flows)}, each of them has more than 2 pkts.')
#     if abnormal:
#         return step_flows
#
#     return flows


def label_flows(flows, pth_label='xxx.csv'):
    """
    1. The number of flows in pth_label is more than flows in pcap, is it possible?
    2. Some flow appears in pth_label, but not in flows, or vice versa, is it possible?

    Parameters
    ----------
    flows
    pth_label

    Returns
    -------

    """
    NORMAL_LABELS = [v.upper() for v in ['benign', 'normal']]
    # ANOMALY_LABELS = [v.upper() for v in ['ANOMALY', 'Malicious', 'FTP-PATATOR', 'SSH-PATATOR',
    #                                       'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
    #                                       'Heartbleed',
    #                                       'Web Attack – Brute Force', 'Web Attack – XSS',
    #                                       'Web Attack – Sql Injection', 'Infiltration',
    #                                       'Bot', 'PortScan', 'DDoS']]

    NORMAL = 'normal'.upper()
    ABNORMAL = 'abnormal'.upper()

    # load CSV with pandas
    csv = pd.read_csv(pth_label)

    labels = {}  # {fid:(1, 0)} # 'normal':1, 'abnormal':0
    cnt_anomaly = 0
    cnt_nomral = 0

    for i, r in enumerate(csv.index):
        if i % 10000 == 0:
            print("Label CSV row {}".format(i))
        row = csv.loc[r]
        # parse 5-tuple flow ID
        # When you merge two csvs with headers, the file includes 'LABEL' means this line is the header
        # so just skip it
        if 'LABEL' in row[" Label"].upper():
            continue
        fid = (str(row[" Source IP"]), str(row[" Destination IP"]), int(row[" Source Port"]),
               int(row[" Destination Port"]), int(row[" Protocol"]))
        # ensure all 5-tuple flows have same label
        label_i = row[" Label"].upper()
        if label_i in NORMAL_LABELS:
            label_i = NORMAL
            cnt_nomral += 1
        else:
            label_i = ABNORMAL
            cnt_anomaly += 1

        if fid in labels.keys():
            labels[fid][label_i] += 1  # labels = {fid: {'normal':1, 'abnormal': 1}}
        else:
            v = 1 if label_i == NORMAL else 0
            labels[fid] = {NORMAL: v, ABNORMAL: 1 - v}

    # decide the true label of each fid
    conflicts = {}
    mislabels = {NORMAL: 0, ABNORMAL: 0}
    for fid, value in labels.items():
        if value[ABNORMAL] > 0 and value[NORMAL] > 0:
            conflicts[fid] = value

        if value[NORMAL] > value[ABNORMAL]:
            labels[fid] = NORMAL
            mislabels[NORMAL] += value[ABNORMAL]  # label 'abnormal' as 'normal'
        else:
            labels[fid] = ABNORMAL
            mislabels[ABNORMAL] += value[NORMAL]  # label 'normal' as 'abnormal'

    # for debug
    an = 0
    na = 0
    for fid, value in conflicts.items():
        if value[NORMAL] > value[ABNORMAL]:
            an += value[ABNORMAL]
        else:
            na += value[NORMAL]

    print(f'label_csv: cnt_normal: {cnt_nomral}, cnt_anomaly: {cnt_anomaly}, Unique labels: {len(labels.keys())}, '
          f'Counter(labels.values()),{Counter(labels.values())}, conflicts: {len(conflicts.keys())}'
          f', mislabels = {mislabels},  abnormal labeled as normal: {an}, normal labeled as abnormal: {na}')

    # obtain the labels of the corresponding features
    new_labels = []
    not_existed_fids = []
    new_fids = []
    for i, (fid, pkt_time, pkt) in enumerate(flows):
        if i == 0:
            print(f'i=0: fid: {fid}, list(labels.keys())[0]: {list(labels.keys())[0]}')
        if fid in labels.keys():
            new_labels.append(labels[fid])
            new_fids.append(fid)
        else:
            not_existed_fids.append(fid)
            new_fids.append('None')
            new_labels.append('None')  # the fid does not exist in labels.csv

    print(f'***{len(not_existed_fids)} (unique fids: {len(set(not_existed_fids))}) flows do not exist in {pth_label},'
          f'Counter(not_existed_fids)[:10]{list(Counter(not_existed_fids))[:10]}')
    print(f'len(new_labels): {len(new_labels)}, unique labels of new_labels: {Counter(new_labels)}')
    return (new_fids, new_labels)


def choose_flows(flows, labels, experiment='ind', random_state=42, pcap_file=''):
    """ obtain normal and anomaly flows and drop the rest.

    Parameters
    ----------
    flows
    labels
    experiment
    random_state

    Returns
    -------

    """
    # if experiment.upper() in ['INDV', 'MIX']:  # for indv and mix use all of data.
    #     return flows, labels

    cnt_normal = 0
    cnt_anomaly = 0
    others = []
    print(Counter(labels))
    for i, label_i in enumerate(labels):
        if label_i.upper() in ['NORMAL', 'BENIGN']:
            cnt_normal += 1
        elif label_i.upper() in ['BOT', 'ANOMALY', 'MALICIOUS', 'ABNORMAL']:
            cnt_anomaly += 1
        else:
            others.append(label_i)

    print(
        f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}, cnt_others: {len(others)}, Counter(others): '
        f'{Counter(others)}')
    get_all_flows_flg = True  # True: use all of flows samples, False: random select part of flow samples.
    # if 'DS50_MAWI_WIDE' in pcap_file or 'DS40_CTU_IoT' in pcap_file:
    #     CNT_ANOMALY = 400
    if 'DS60_UChi_IoT' in pcap_file or 'DS20_PU_SMTV' in pcap_file:
        CNT_ANOMALY = 600
    elif 'DS10_UNB_IDS' in pcap_file:
        if cnt_normal > int(7 / 3 * cnt_anomaly):  # normal: anomaly = 7:3
            CNT_ANOMALY = cnt_anomaly
        else:
            CNT_ANOMALY = int(cnt_normal * 3 / 7)
    else:
        CNT_ANOMALY = 400

    if cnt_anomaly < CNT_ANOMALY:
        part_anomaly_thres = cnt_anomaly
    else:
        part_anomaly_thres = CNT_ANOMALY
    if cnt_anomaly < 10:
        print(f'skip cnt_anomaly(={part_anomaly_thres}) < 10')
        print(f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}=> part_anomaly_thres: {part_anomaly_thres}')
        return -1

    part_normal_thres = 5000 + part_anomaly_thres  # only random get 20000 normal samples.
    if cnt_normal > part_normal_thres or cnt_anomaly > part_anomaly_thres:
        get_all_flows_flg = False  # make all data have the same size
        # break # if has break here, it only print part of flows in cnt_normal

    print(f'before, len(flows): {len(flows)}, len(lables): {len(labels)}, get_all_flows_flg: {get_all_flows_flg}, '
          f'cnt_normal: {cnt_normal}, cnt_anomaly: {cnt_anomaly}')

    if not get_all_flows_flg:
        c = list(zip(flows, labels))
        flows_shuffle, labels_shuffle = zip(*shuffle(c, random_state=random_state))
        cnt_normal = 0
        cnt_anomaly = 0
        flows = []
        labels = []
        for i, (flows_i, label_i) in enumerate(zip(flows_shuffle, labels_shuffle)):
            if label_i.upper() in ['NORMAL', 'BENIGN']:
                cnt_normal += 1
                if cnt_normal <= part_normal_thres:
                    flows.append(flows_i)
                    labels.append(label_i)
            elif label_i.upper() in ['BOT', 'ANOMALY', 'MALICIOUS']:
                cnt_anomaly += 1
                if cnt_anomaly <= part_anomaly_thres:
                    flows.append(flows_i)
                    labels.append(label_i)

            if cnt_normal > part_normal_thres and cnt_anomaly > part_anomaly_thres:
                break
        else:
            pass
    print(f'after: len(flows): {len(flows)}, len(lables): {len(labels)}, get_all_flows_flg: {get_all_flows_flg}, '
          f'cnt_normal: {min(cnt_normal, part_normal_thres)}, cnt_anomaly: {min(cnt_anomaly, part_anomaly_thres)}')

    return flows, labels


def copy_file(src, dst):
    """ Copy file

    Parameters
    ----------
    src
    dst

    Returns
    -------

    """
    out_dir = os.path.dirname(dst)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    copyfile(src, dst)


def extract_subpcap(pcap_file, out_file, start_time, end_time, verbose=20, keep_original=True):
    """ extract a part of pcap using editcap
    ' editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" input.pcap output.pcap'

    Parameters
    ----------
    pcap_file:
    out_file
    start_time
    end_time
    verbose
    keep_original: bool
        keep the original pcap or not, True (default)

    Returns
    -------

    """

    if not os.path.exists(out_file):
        out_file = pcap_file + f'-start={start_time}-end={end_time}.pcap'
        out_file = out_file.replace(' ', '_')
        if os.path.exists(out_file): return out_file

    cmd = f"editcap -A \"{start_time}\" -B \"{end_time}\" {pcap_file} {out_file}"
    # print(cmd)
    if verbose > 10:
        print(f'{cmd}')
    result = ''
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        if not keep_original:
            os.remove(pcap_file)
    except Exception as e:
        print(f'{e}, {result}')

    return out_file


def filter_ip(pcap_file, out_file, ips=[], direction='src_dst', keep_original=True, verbose=20):
    if not os.path.exists(pcap_file): return ''
    if not pth.exists(pth.dirname(out_file)):
        os.makedirs(pth.dirname(out_file))

    if direction == 'src':
        ip_str = " or ".join([f'ip.src=={ip}' for ip in ips])
    elif direction == 'dst':
        ip_str = " or ".join([f'ip.dst=={ip}' for ip in ips])
    else:  # src_dst, use forward + backward data
        ip_str = " or ".join([f'ip.addr=={ip}' for ip in ips])
    cmd = f"tshark -r {pcap_file} -w {out_file} {ip_str}"

    if verbose > 10: print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        if not keep_original:
            os.remove(pcap_file)
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def filter_csv_ip(label_file, out_file, ips=[], direction='src_dst', keep_original=True, verbose=10):
    # from shutil import copyfile
    # copyfile(label_file, out_file)

    # print(label_file_lst, mrg_label_path)
    # if os.path.exists(mrg_label_path):
    #     os.remove(mrg_label_path)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with open(out_file, 'w') as out_f:
        header = True
        with open(label_file, 'r') as in_f:
            line = in_f.readline()
            while line:
                if line.strip().startswith('Flow ID') and header:
                    if header:
                        header = False
                        print(line)
                        out_f.write(line.strip('\n') + '\n')
                    else:
                        pass
                    line = in_f.readline()
                    continue
                if line.strip() == '':
                    line = in_f.readline()
                    continue

                exist = False
                for ip in ips:
                    if ip in line:
                        exist = True
                        break
                if exist:
                    out_f.write(line.strip('\n') + '\n')
                line = in_f.readline()

    return out_file


@execute_time
def load_flows_label_pickle(output_flows_labels):
    with open(output_flows_labels, 'rb') as in_hdl:
        flows, labels, subflow_interval = pickle.load(in_hdl)

    return flows, labels, subflow_interval
