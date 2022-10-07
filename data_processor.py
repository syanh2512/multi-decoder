from datetime import datetime
from torch import LongTensor
from torch.nn.functional import cosine_similarity
import numpy
import torch
import time

class Preprocessor:
    def __init__(self, hidden_vector_dict, word2idx):
        self.hidden_vector_dict = hidden_vector_dict
        self.word2idx = word2idx

    def generate(self, time, duration, protocol, src_ip, src_port, dst_ip, dst_port, packets, bytes, flags):
        self.numeric = numpy.array([
        #   isMon   isTue   isWed   isThu   isFri   isSat   isSun   daytime
            0,      0,      0,      0,      0,      0,      0,      0.,
        #   isTCP   isUDP   isICMP
            0,      0,      0,
        #   isURG   isACK   isPSH   isRES   isSYN   isFIN
            0,      0,      0,      0,      0,      0,
        ])

        self.get_time_attr(time)
        self.get_proto_attr(protocol)
        self.get_flags_attr(flags)
        self.vector = self.get_vector(duration, src_ip, src_port, dst_ip, dst_port, packets, bytes)
        return numpy.concatenate((self.numeric, self.vector), axis=0)

    def get_time_attr(self, date_seen):
        time = datetime.strptime(date_seen, '%Y-%m-%d %H:%M:%S.%f')
        day = time.weekday()

        if      day == 0:   self.numeric[0] = 1
        elif    day == 1:   self.numeric[1] = 1
        elif    day == 2:   self.numeric[2] = 1
        elif    day == 3:   self.numeric[3] = 1
        elif    day == 4:   self.numeric[4] = 1
        elif    day == 5:   self.numeric[5] = 1
        elif    day == 6:   self.numeric[6] = 1

        self.numeric[7] = (
                    time.second 
                +60*time.minute 
              +3600*time.hour
            )/86400

    def get_proto_attr(self, protocol):
        if      protocol == 'TCP':    self.numeric[8] = 1
        elif    protocol == 'UDP':    self.numeric[9] = 1
        elif    protocol == 'ICMP':   self.numeric[10] = 1
        
    def get_flags_attr(self, flags):
        flags = list(flags)
        if  flags[0] == 'U':    self.numeric[11] = 1
        if  flags[1] == 'A':    self.numeric[12] = 1
        if  flags[2] == 'P':    self.numeric[13] = 1
        if  flags[3] == 'R':    self.numeric[14] = 1
        if  flags[4] == 'S':    self.numeric[15] = 1
        if  flags[5] == 'F':    self.numeric[16] = 1

    def get_vector(self, duration, src_ip, src_port, dst_ip, dst_port, packets, bytes):
        du_idx = self.word2idx["Duration"][duration]
        src_ip_idx = self.word2idx["IP address"][src_ip]
        src_pt_idx = self.word2idx["Port"][src_port]
        dst_ip_idx = self.word2idx["IP address"][dst_ip]
        dst_pt_idx = self.word2idx["Port"][dst_port]
        pk_idx = self.word2idx["Number of transmitted packets"][packets]
        bt_idx = self.word2idx["Number of transmitted bytes"][bytes]

        du_vec = self.hidden_vector_dict[du_idx].cpu().detach().numpy().reshape(-1)
        src_ip_vec = self.hidden_vector_dict[src_ip_idx].cpu().detach().numpy().reshape(-1)
        src_pt_vec = self.hidden_vector_dict[src_pt_idx].cpu().detach().numpy().reshape(-1)
        dst_ip_vec = self.hidden_vector_dict[dst_ip_idx].cpu().detach().numpy().reshape(-1)
        dst_pt_vec = self.hidden_vector_dict[dst_pt_idx].cpu().detach().numpy().reshape(-1)
        pk_vec = self.hidden_vector_dict[bt_idx].cpu().detach().numpy().reshape(-1)
        bt_vec = self.hidden_vector_dict[pk_idx].cpu().detach().numpy().reshape(-1)

        vector = numpy.concatenate((du_vec, src_ip_vec, src_pt_vec, dst_ip_vec, dst_pt_vec, bt_vec, pk_vec), axis=0)

        return vector.flatten()
