"""
Classes have been added to this file for cluster training.
Also Lockdown is a transformer.
"""
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
d_model = 256
num_heads = 4
dropout = 0.1

class WCST:
    
    def __init__(self, batch_size):
        self.colours = ['red','blue','green','yellow']
        self.shapes = ['circle','square','star','cross']
        self.quantities = ['1','2','3','4']
        self.categories = ['C1','C2','C3','C4']
        self.category_feature = np.random.choice([0,1,2])
        self.gen_deck()
        self.batch_size = batch_size

    def gen_deck(self):
        cards = []
        for colour in self.colours:
            for shape in self.shapes:
                for quantity in self.quantities:
                    cards = cards + [(colour, shape, quantity)]
        self.cards = np.array(cards)
        self.card_indices = np.arange(len(cards))

    def context_switch(self):
        self.category_feature = np.random.choice(np.delete([0,1,2],self.category_feature))

    def gen_batch(self):
        batch_size = self.batch_size
        while True:
            prev_feature = self.category_feature
            category_level = np.abs(self.category_feature - 2)+1
            card_partitions = [np.concatenate([np.arange(4**(category_level-1)) + feature_value*(4**(category_level-1))
                              + start for start in np.arange(0,64,int(4**(category_level)))])
                              for feature_value in range(4)]
            category_cards = np.vstack([np.random.choice(card_partition, batch_size, replace=True) for card_partition in card_partitions]).T
            category_cards = category_cards[np.arange(batch_size)[:,np.newaxis], [np.random.permutation(4) for _ in range(batch_size)]]
            category_cards_feature = (category_cards % (4**category_level)) // (4**(category_level-1))
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (category_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 60)
            example_cards = available_cards[np.arange(batch_size),np.random.randint(0,60,(batch_size))]
            example_cards_feature = (example_cards % (4**category_level)) // (4**(category_level-1))
            example_labels = np.argmin(np.abs(category_cards_feature - example_cards_feature[:,np.newaxis]), axis=1)
            used_cards = np.hstack([category_cards,example_cards[:,np.newaxis]]).astype(int)
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (used_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 59)
            question_cards = available_cards[np.arange(batch_size),np.random.randint(0,59,(batch_size))]
            question_cards_feature = (question_cards % (4**category_level)) // (4**(category_level-1))
            question_labels = np.argmin(np.abs(category_cards_feature - question_cards_feature[:,np.newaxis]),axis=1)
            yield np.hstack([category_cards,example_cards[:,np.newaxis],np.ones((batch_size,1))*68,\
                          example_labels[:,np.newaxis]+64,np.ones((batch_size,1))*69]),\
                   np.hstack([question_cards[:,np.newaxis],np.ones((batch_size,1))*68,question_labels[:,np.newaxis]+64])

    def visualise_batch(self,batch):
        trials = []
        batch = np.hstack(batch)
        for trial_idx in range(batch.shape[0]):
            trial = batch[trial_idx].astype(int)
            trial_cards = []
            for token_idx in trial:
                if token_idx < 64:
                    trial_cards = trial_cards + [self.cards[token_idx]]
                elif token_idx < 68:
                    trial_cards = trial_cards + [self.categories[token_idx-64]]
                elif token_idx == 68:
                    trial_cards = trial_cards + ['SEP']
                elif token_idx == 69:
                    trial_cards = trial_cards + ['EOS']
            trials = trials + [trial_cards]
            print(trial_cards)
        print("Feature for Classification: ", self.category_feature, "\n")
        return trials

class Dataset_Loader:
    def __init__(self, training_batch, classification_batch, train_split, val_split, test_split, context_switch_interval):
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.classification_batch = classification_batch
        self.training_batch = training_batch
        self.context_switch_interval = context_switch_interval

    def _trial_key(self, input_batch, target_batch):
        """
        Create a unique key per trial including:
        - category cards
        - example card
        - example label
        - question card
        - question label (target)
        """
        keys = []
        for i in range(input_batch.shape[0]):
            cat_cards = tuple(input_batch[i, :4])
            example_card = input_batch[i, 4]
            example_label = input_batch[i, 5]
            question_card = input_batch[i, 6]
            question_label = target_batch[i, 0]
            keys.append((cat_cards,
                         example_card,
                         example_label,
                         question_card,
                         question_label))
        return keys

    def load_data(self):
        """
        The sum of train, val and test splits should equal training_batch.
        Ensures no duplicate trials across train/val/test.
        """
        wcst_env = WCST(self.classification_batch)
        train_data = []
        val_data = []
        test_data = []
        seen = set()  # track all seen trial keys

        n_train = int(np.floor(self.training_batch * self.train_split))
        n_val = int(np.floor(self.training_batch * self.val_split))
        n_test = int(np.floor(self.training_batch * self.test_split))

        def fill_dataset(n_items, dataset_list, start_count=0):
            count = 0
            while count < n_items:
                if (start_count + count) % self.context_switch_interval == 0 and (start_count + count) != 0:
                    wcst_env.context_switch()

                input_batch, target_batch = next(wcst_env.gen_batch())
                keys = self._trial_key(input_batch, target_batch)

                new_indices = [i for i, k in enumerate(keys) if k not in seen]
                if len(new_indices) == 0:
                    continue  # skip this batch 

                for i in new_indices:
                    seen.add(keys[i])
                    dataset_list.append((input_batch[i:i+1], target_batch[i:i+1]))
                    count += 1
                    if count >= n_items:
                        break
            return count

        # fill each split ensuring uniqueness
        c1 = fill_dataset(n_train, train_data, start_count=0)
        c2 = fill_dataset(n_val, val_data, start_count=c1)
        c3 = fill_dataset(n_test, test_data, start_count=c1+c2)

        return train_data, val_data, test_data

class SelfAttention(nn.Module):
    """
    Note:
        input vectors: X
        query vectors: Q = XW(q), where W(q) is the query weight matrix
        key vectors: K = XW(k), where W(k) is the key weight matrix
        value vectors: V = XW(v), where W(v) is the value weight matrix
        N(_) -> number of vectors
        D(_) -> dimension of vectors

        input_vectors -> Input embeddings shape(N(X) x D(X))
        key_matrix -> Key weight matrix shape(D(X) x D(Q))
        value_matrix -> Value weight matrix shape(D(X) x D(V))
        query_matrix -> Query weight matrix shape(D(X) x D(Q))
    """
    def __init__(self, embed_dim, qkv_dim):
        """
        Args:
            embed_dim: size of input embeddings (D(X))
            qkv_dim: size of Q, K, V vectors (D(Q)=D(K)=D(V))
        """
        super().__init__()
        # Learnable weight matrices
        self.W_q = nn.Linear(embed_dim, qkv_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, qkv_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, qkv_dim, bias=False)
        
    def scaled_dot_product(self,query, key, query_dim):
        scaling_factor = 1 / np.sqrt(query_dim)
        product = torch.einsum('bqd,bkd->bqk', query, key) #tensor product
        return scaling_factor * product

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        Q = self.W_q(x)  # shape(N(X) x D(Q))
        K = self.W_k(x)    # shape(N(X) x D(K))
        V = self.W_v(x)  # shape(N(X) x D(V))

        similarity = self.scaled_dot_product(Q, K, Q.shape[-1])  # shape(N(X) x N(X))
        attention_weights = torch.softmax(similarity, dim=-1)  # shape(N(X) x N(X)) softmax along columns
        output = torch.matmul(attention_weights, V)  # shape(N(X) x D(V))
        return output
        
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert d_model % num_heads == 0, "divisible"
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        def shape(self, x: torch.Tensor) -> torch.Tensor:
            B, T, _ = x.size()
            return x.view(B, T, self.h, self.dh).transpose(1, 2)
        
        def mask(self, Tq: int, Tk: int, device) -> torch.Tensor:
            i = torch.arange(Tq, device=device).unsqueeze(1)
            j = torch.arange(Tk, device=device).unsqueeze(0)
            return (j <= i)  


    def forward(self,query: torch.Tensor, key: torch.Tensor,value: torch.Tensor, 
                attn_mask: torch.Tensor = None,
                 additive_mask: torch.Tensor = None,
                     causal: bool = False,):
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()

        q = self._shape(self.Wq(query))   
        k = self._shape(self.Wk(key))     
        v = self._shape(self.Wv(value))   

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5) 

        if causal:
            assert Tq == Tk, "causal mask expects square (self-attn) lengths"
            causal_keep = self._causal_mask(Tq, Tk, query.device)              
            attn_logits = attn_logits.masked_fill(~causal_keep.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))

        if additive_mask is not None:
            attn_logits = attn_logits + additive_mask  

        attn_weights = torch.softmax(attn_logits, dim=-1)  
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)            

        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  
        out = self.Wo(context)  
        return out, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model]
        """
        pass

class Embedding(nn.Module):
    def __init__(self):
        """
        Args:
       
            
        returns: [embeddings] as tensor
        """
        pass

    def forward(self, x):
        pass

class PositionalEncoder:
    """
    Sinusoidal Positional Encoder
    """
    @staticmethod
    def encode(input_seq, model_dim):
        """
        Args:
            input_seq -> The whole input sequence [batch, seq_len]
            model_dim -> The model dimension(hyperparameter)
            
        returns: [positional encodings] as tensor
        """
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        pe = np.zeros((batch_size, seq_len, model_dim))

        for pos in range(seq_len):
            for i in range(0, model_dim, 2):
                theta = pos / (10000 ** (2 * i / model_dim))
                pe[:, pos, i] = np.sin(theta)
                pe[:, pos, i+1] = np.cos(theta)

        return torch.tensor(pe, dtype=torch.float32)
  
class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.drop = nn.Dropout(dropout)

    def _expand_key_mask(self, pad_mask: torch.Tensor, Tq: int, heads: int) -> torch.Tensor:
        
        B, Tk = pad_mask.size()
        keep = pad_mask.bool().unsqueeze(1).unsqueeze(2)        
        keep = keep.expand(B, heads, Tq, Tk)                    
        return keep

    def forward(self, x, enc_out, tgt_pad=None, src_pad=None):
        attn_maps = {}

        B, T_tgt, _ = x.size()
        self_keep = None
        if tgt_pad is not None:
            self_keep = self._expand_key_mask(tgt_pad, T_tgt, self.self_attn.h)

        x_norm = self.ln_self(x)
        self_out, self_w = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=self_keep,  
            causal=True
        )
        x = x + self.drop(self_out)
        attn_maps["self"] = self_w  

        B, T_src, _ = enc_out.size()
        cross_keep = None
        if src_pad is not None:
            cross_keep = self._expand_key_mask(src_pad, T_tgt, self.cross_attn.h)

        x_norm = self.ln_cross(x)
        cross_out, cross_w = self.cross_attn(
            query=x_norm, key=enc_out, value=enc_out,
            attn_mask=cross_keep,  
            causal=False
        )
        x = x + self.drop(cross_out)
        attn_maps["cross"] = cross_w  

        return x, attn_maps

class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len=5000):
        super().__init__()
        # Encoder
        # Decoder
        # Final linear layer to project to target vocab
        # output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch, src_seq_len]
        tgt: [batch, tgt_seq_len]
        returns: logits over target vocab [batch, tgt_seq_len, vocab_size]
        """
        pass

if __name__ == "__main__":
    data = Dataset_Loader(training_batch=10,
                          classification_batch=2,
                            train_split=0.8,
                            val_split=0.10,
                            test_split=0.10,
                            context_switch_interval=2)
    train_data, val_data, test_data = data.load_data()

    positional_encodings = PositionalEncoder.encode(
        input_seq=train_data[0][0], model_dim=10)
    print("Positional Encodings Shape: ", positional_encodings.shape)
    print("positional encodings: ", positional_encodings[0])
