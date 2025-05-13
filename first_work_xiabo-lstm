'''目前效果最好的 小波-lstm'''
class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4):
        super(Module, self).__init__()
        # 设置随机种子
        self.setup_seed(0)
        """multi-view feature extractor"""
        self.text_view_extract=ASR_model()
        self.emo_view_extract=SER_model()
        # Embedding adjustment layers
        self.embedding_adjustment_audio = nn.Linear(embed_dim, 256)
        self.embedding_adjustment_emo = nn.Linear(embed_dim, 256)

        # LSTM layer
        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True,bidirectional=True)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True)

        # Attention layer
        self.my_attention = MyAttention(embed_dim=256, num_heads=4)
        self.dense_1 = nn.Linear(256, 128)
        self.batch_normalization_1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense_2 = nn.Linear(128, 64)
        self.batch_normalization_2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.dense_3 = nn.Linear(64, 2)

        # Define multiple Block layers
        self.blocks = nn.ModuleList([Block(dim=embed_dim) for _ in range(depth)])

        # Fully connected layers for final classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, inputs,inputs2, Freq_aug):
        x=inputs
        x2=inputs2
        """multi-view features"""
        emo_view=self.emo_view_extract(x2) # 10 50 64  (bs,frame_number,feat_out_dim)
        # print(emo_view.shape)
        text_view=self.text_view_extract(x) # 10 50 64
        # print(text_view.shape)
        concat_view = torch.cat([text_view,emo_view], dim=-1) # 10 50 128
        concat_view = concat_view.transpose(1, 2)

        for blk in self.blocks:
            x = blk(concat_view)

        x = x.permute(0, 2, 1) #[1, 67, 128] 每个时间步有128个特征
        # Adjust embedding dimension for LSTM
        x = self.embedding_adjustment_audio(x)
        # Process through LSTM
        x, _ = self.lstm(x)  # [batch_size, frame_number, hidden_size]
        # Apply attention mechanism
        x = x.transpose(0, 1)  # [frame_number, batch_size, hidden_size]
        x = self.my_attention(x)  # [frame_number, batch_size, hidden_size]
        x = x.transpose(0, 1)  # [batch_size, frame_number, hidden_size]
        # Select the last time step
        x = x[:, -1, :]  # [batch_size, hidden_size]

        x = self.dense_1(x)
        x = self.batch_normalization_1(x)
        x = self.dropout1(x)
        x = self.dense_2(x)
        x = self.batch_normalization_2(x)
        x = self.dropout2(x)
        x = self.dense_3(x)
        return x
