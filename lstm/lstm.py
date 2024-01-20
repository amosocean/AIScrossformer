import torch
import torch.nn as nn

class BasicLSTMNet(nn.Module):
    """     
    一个没有形状限制的LSTM模块,没有Linear层用于各个layer的输出综合
     """
    #def __init__(self, input_size=9, hidden_size=64, num_layers=4, output_size=5):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BasicLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)#输入(N,Length,Input_channel)顺序
    def forward(self, x,h0=None,c0=None):
        if x.ndim == 2:
            #x=x.unsqueeze(dim=0)
            assert x.ndim == 3 ,"LSTM Input dimension Error"
        x=x.transpose(-2,-1)#调整为(N,Length,Input_channel)顺序 (project原顺序 [...,batch,xDim(feature),TimeDim(length)])
        # 初始化 LSTM 隐藏状态
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device="cuda",requires_grad=True)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device="cuda",requires_grad=True)
        # 前向传递 LSTM
        out, (hn, cn)= self.lstm(x, (h0, c0))
        #rtn=torch.nested.as_nested_tensor([out,hn,cn]) #按照官方文档的说法，这一步复制了张量
        out=out.transpose(-2,-1) #调整为project原顺序 [...,batch,xDim(feature),TimeDim(length)]
        return out, (hn, cn)

class LSTM_EncoderDecoder_Common(nn.Module):
    """     
    Seq2Seq 的LSTM编解码器,输入输出长度可以任意变化。解码过程for循环
    project函数共用一个线性层
    """
    def __init__(self,input_length:int, input_size:int, hidden_size:int, num_layers:int,output_length:int,output_size:int):
        super(LSTM_EncoderDecoder_Common,self).__init__()
        params=locals()
        del params["self"]
        vars(self).update(params)
        
        self.encoder=BasicLSTMNet(input_size, hidden_size, num_layers)
        self.decoder=BasicLSTMNet(hidden_size,hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    
    def project_base(self,decoder_output:torch.Tensor)->torch.Tensor:
        #dim=-2
        decoder_output=decoder_output.squeeze(dim=-1)
        input_size=self.hidden_size
        output_size=self.output_size
        fc=nn.Linear(input_size, output_size, device=decoder_output.device)
        #decoder_output=decoder_output.transpose(dim,-1)
        decoder_output_projected=fc(decoder_output)
        #decoder_output=decoder_output.transpose(dim,-1)
        return decoder_output_projected
    
    
    def forward(self, input_seq):
            input_seq = input_seq.transpose(-2,-1)
            # 编码器
            encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input_seq)

            # 解码器
            decoder_input = encoder_hidden[-1]
            decoder_input = decoder_input.unsqueeze(dim=-1)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell
            decoder_outputs = []
            for i in range(self.output_length):  # 解码器预测self.output_length个token
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, h0=decoder_hidden,c0=decoder_cell)
                
                decoder_output_projected=decoder_output.transpose(-2,-1) #调整为(N,Length,Input_channel)顺序 (project原顺序 [...,batch,xDim(feature),TimeDim(length)])
                decoder_output_projected = self.linear(decoder_output_projected)
                decoder_output_projected=decoder_output_projected.transpose(-2,-1) #调整回去[...,batch,xDim(feature),TimeDim(length)]

                decoder_outputs.append(decoder_output_projected)
                decoder_input = decoder_output
            decoder_outputs_projected = torch.stack(decoder_outputs, dim=-1).squeeze(-2)#[...,batch,xDim(feature),1,]->[...,batch,xDim(feature),1,TimeDim(length)]->[...,batch,xDim(feature),TimeDim(length)]
            
            return decoder_outputs_projected.transpose(-2,-1)