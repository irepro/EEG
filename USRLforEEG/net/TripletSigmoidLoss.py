
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

class TripletSigmoidLoss(torch.nn.modules.loss._Loss):
    def __init__(self, Kcount = 5, sample_margin =10 ):
        super().__init__()
        self.Kcount = Kcount
        self.margin = sample_margin

    def forward(self, batch, encoder, train, **kwargs):
        train_size = train.size(0)
        max_length = train.size(1)
        batch_size = batch.size(0)

        samples = np.random.choice(
                    train_size, size=(self.Kcount, batch_size),replace=False
                )
        samples = torch.LongTensor(samples)

        lengths_batch = max_length - torch.sum(
                        torch.isnan(batch[:,0]), 1
                    ).data.cpu().numpy()
        lengths_samples = np.empty(
                        (self.Kcount, batch_size), dtype=int
                    )
        for i in range(self.Kcount):
            lengths_samples[i] = max_length - torch.sum(
                torch.isnan(train[samples[i],0]), 1
            ).data.cpu().numpy()

        '''print("lengths_samples :\n", lengths_samples)
        print("lengths_batch :\n", lengths_batch)'''

        lengths_pos = np.empty(batch_size, dtype=int)
        lengths_neg = np.empty(
            (self.Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                self.margin, lengths_batch[j] + 1
            )
            for i in range(self.Kcount):
                lengths_neg[i, j] = np.random.randint(
                    self.margin, lengths_samples[i, j] + 1
                )

        '''print("lengths_pos :\n", lengths_pos)'''

        beginning_pos = np.empty(batch_size, dtype=int)
        beginning_neg = np.empty(
            (self.Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            beginning_pos[j] = np.random.randint(
                0, lengths_batch[j] - lengths_pos[j] + 1
            )
            for i in range(self.Kcount):
                beginning_neg[i, j] = np.random.randint(
                    0, lengths_samples[i, j] - lengths_neg[i, j] + 1
                )

        '''print("begin_pos :\n", beginning_pos)'''

        lengths_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            lengths_ref[j] = np.random.randint(
                self.margin, lengths_pos[j] + 1
            )
        beginning_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            beginning_ref[j] = np.random.randint(
                beginning_pos[j], lengths_pos[j] - lengths_ref[j] + beginning_pos[j] + 1
            )

        '''print("length_ref :\n", lengths_ref)
        print("begin_ref :\n", beginning_ref)'''

        ref = []
        for j in range(batch_size):
            ref.append(encoder.forward(batch[j,:,beginning_ref[j]:beginning_ref[j]+lengths_ref[j]]))
        ref = torch.stack(ref)
        pos = []
        for j in range(batch_size):
            pos.append(encoder.forward(batch[j,:,beginning_pos[j]:beginning_pos[j]+lengths_pos[j]]))
        pos = torch.stack(pos)
        neg = []
        for j in range(batch_size):
            neg_tensor = [encoder.forward(train[samples[i,j],:,beginning_neg[i,j]:beginning_neg[i,j]+lengths_neg[i,j]]) for i in range(self.Kcount)]
            neg.append(torch.stack(neg_tensor))
        neg = torch.stack(neg)

        batch_size = batch.size(0)
        size_representation = encoder.out_channels
        
        loss, loss_tr = 0, 0
        
        loss += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(ref,pos.view(batch_size, size_representation, 1)))) 
        loss_tr += loss

        loss.backward(retain_graph=True)
        loss = 0
        del pos
        torch.cuda.empty_cache()

        for i in range(self.Kcount):
            neg_tensor = neg[:,i,:,:]
            loss += -torch.mean(torch.nn.functional.logsigmoid(
            -torch.bmm(ref,neg_tensor.view(batch_size, size_representation, 1))))
            
        loss_tr += loss
        loss.backward(retain_graph=True)
        loss = 0
        del neg, ref
        torch.cuda.empty_cache()

        return loss_tr

    def get_valloss(self,  encoder, val, **kwargs):
        train_size = val.size(0)
        max_length = val.size(1)

        samples = np.random.choice(
                    train_size, size=(self.Kcount+1),replace=False
                )
        samples = torch.LongTensor(samples)

        lengths_samples = np.empty(
                        (self.Kcount+1), dtype=int
                    )
        for i in range(self.Kcount+1):
            lengths_samples[i] = max_length - torch.sum(
                torch.isnan(val[samples[i],0]), 0
            ).data.cpu().numpy()

        lengths_pos = np.random.randint(
                self.margin, lengths_samples[0] + 1
            )
        lengths_neg = np.empty(
            (self.Kcount), dtype=int
        )
        for j in range(self.Kcount):
            lengths_neg[j] = np.random.randint(
                self.margin, lengths_samples[j+1] + 1
            )

        beginning_pos = np.random.randint(
                0, lengths_samples[0] - lengths_pos + 1
            )
        beginning_neg = np.empty(
            (self.Kcount), dtype=int
        )
        for j in range(self.Kcount):
            beginning_neg[j] = np.random.randint(
                0, lengths_samples[j+1] - lengths_neg[j] + 1
            )

        lengths_ref = np.random.randint(
                self.margin, lengths_pos + 1
            )
        beginning_ref = np.random.randint(
                beginning_pos, lengths_pos - lengths_ref + beginning_pos + 1
            )

        ref = encoder.forward(val[samples[0],:, beginning_ref:beginning_ref+lengths_ref])
        pos = encoder.forward(val[samples[0],:, beginning_pos:beginning_pos+lengths_pos])
        neg = []
        for j in range(self.Kcount):
            neg_tensor = encoder.forward(val[samples[j+1],:, beginning_neg[j]:beginning_neg[j]+lengths_neg[j]])
            neg.append(neg_tensor)
        neg = torch.stack(neg)

        loss = 0

        loss += -torch.nn.functional.logsigmoid(torch.matmul(ref,pos.view(-1,1)))

        for i in range(self.Kcount):
            neg_tensor = neg[i,:,:]
            loss += -torch.mean(torch.nn.functional.logsigmoid(
            -torch.matmul(ref,neg_tensor.view(-1,1))))
         
        del pos, neg, ref
        torch.cuda.empty_cache()

        return loss

    
    '''
    def gettripletsigLoss(input, batch, Kcount, device):
        batch_size = batch.size(0)
        ref, pos, neg = input
        size_representation = ref.size(1)
        

        loss = 0
        
        loss += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(ref.view(batch_size, 1, size_representation),
        pos.view(batch_size, size_representation, 1)))) 

        loss.backward(retain_graph=True)
        loss = 0
        del pos
        torch.cuda.empty_cache()

        for i in range(Kcount):
            loss += -torch.mean(torch.nn.functional.logsigmoid(
            -torch.bmm(ref.view(batch_size, 1, size_representation),neg[i].view(batch_size, size_representation, 1))))
            loss.backward(retain_graph=True)
            loss = 0
            del neg
            torch.cuda.empty_cache()
        return loss

    def getRepresent(train, batch, Kcount, margin, encoder):
        train_size = train.size(0)
        max_length = train.size(1)
        batch_size = batch.size(0)

        samples = np.random.choice(
                    train_size, size=(Kcount, batch_size),replace=False
                )
        samples = torch.LongTensor(samples)

        lengths_batch = max_length - torch.sum(
                        torch.isnan(batch[:]), 1
                    ).data.cpu().numpy()
        lengths_samples = np.empty(
                        (Kcount, batch_size), dtype=int
                    )
        for i in range(Kcount):
            lengths_samples[i] = max_length - torch.sum(
                torch.isnan(train[samples[i]]), 1
            ).data.cpu().numpy()

        print("lengths_samples :\n", lengths_samples)
        print("lengths_batch :\n", lengths_batch)

        lengths_pos = np.empty(batch_size, dtype=int)
        lengths_neg = np.empty(
            (Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                margin, lengths_batch[j] + 1
            )
            for i in range(Kcount):
                lengths_neg[i, j] = np.random.randint(
                    margin, lengths_samples[i, j] + 1
                )

        print("lengths_pos :\n", lengths_pos)

        beginning_pos = np.empty(batch_size, dtype=int)
        beginning_neg = np.empty(
            (Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            beginning_pos[j] = np.random.randint(
                0, lengths_batch[j] - lengths_pos[j] + 1
            )
            for i in range(Kcount):
                beginning_neg[i, j] = np.random.randint(
                    0, lengths_samples[i, j] - lengths_neg[i, j] + 1
                )

        print("begin_pos :\n", beginning_pos)

        lengths_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            lengths_ref[j] = np.random.randint(
                margin, lengths_pos[j] + 1
            )
        beginning_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            beginning_ref[j] = np.random.randint(
                beginning_pos[j], lengths_pos[j] - lengths_ref[j] + beginning_pos[j] + 1
            )

        print("length_ref :\n", lengths_ref)
        print("begin_ref :\n", beginning_ref)

        ref_tensor = torch.cat([encoder.forward(batch[j, beginning_ref[j]:beginning_ref[j]+lengths_ref[j]]) for j in range(batch_size)])
        ref_tensor.reshape([batch_size,-1])
        pos_tensor = torch.cat([encoder.forward(batch[j, beginning_pos[j]:beginning_pos[j]+lengths_pos[j]]) for j in range(batch_size)])
        pos_tensor.reshape([batch_size,-1])
        neg_tensor = torch.cat([encoder.forward(batch[j, beginning_neg[j]:beginning_neg[j]+lengths_neg[j]]) for j in range(batch_size)])
        neg_tensor.reshape([batch_size,Kcount,-1])

        represent = [ref_tensor, pos_tensor, neg_tensor]
        return represent
        
        train_size = train.size(1)
        max_length = train.size(0)
        batch_size = batch.size(0)

        samples = numpy.random.choice(
                train_size, size=(, batch)
            )
        
        train.transpose([batch, max_length])
        
        np_length_size = np.empty(batch, dtype=int)
        np_finishindex = np.empty(batch, dtype=int)
        np_beginindex = np.random.choice(max_length, batch)
        diff = max_length - np.beginindex
        for i in range(batch):
            np_length_size[i] = np.random.choice(diff[i], 1)
        np_finishindex = np_beginindex + np_length_size

        spindex = np.random.choice(batch, 6, replace=False)

        ref_beginindex = np.random.choice(np_length_size[spindex[0]], 1)
        diff = np_length_size[spindex[0]] - ref_beginindex
        ref_length_size = max(np.random.choice(diff, 1), margin-1)
        ref_beginindex += np_beginindex[spindex[0]]
        ref_finishindex = ref_beginindex + ref_length_size

        input = []
        for i in spindex:
            if i == spindex[0]:
                input.append([i, ref_beginindex, ref_finishindex, ref_length_size])
            input.append([i, np_beginindex[i], np_finishindex[i], np_length_size[i]])'''

def sigmoid(x):
    return 1 / (1 +np.exp(-x))