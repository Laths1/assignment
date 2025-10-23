
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class RuleDetector:
    def __init__(self, model, test_data, wcst_env, batch_size=32):
        self.model = model
        self.model.eval()
        self.test_loader = DataLoader(self._prepare_data(test_data), batch_size=batch_size)
        self.wcst_env = wcst_env
        self.num_heads = model.decoders[0].cross_attn.h
        self.num_layers = len(model.decoders)
        self.features = ['Colour', 'Shape', 'Quantity']

    def _prepare_data(self, data):
        inputs = np.vstack([d[0] for d in data])
        targets = np.vstack([d[1] for d in data])
        return TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long)
        )

    def _get_match_positions(self, src_batch):
        match_positions_batch = []

        CAT_CARD_POSITIONS = torch.arange(4) 
        
        for i in range(src_batch.size(0)):
            src_i = src_batch[i].cpu().numpy()

            example_card_idx = src_i[4] 

            example_features = self.wcst_env.get_card_features(example_card_idx)

            match_indices = {'Colour': [], 'Shape': [], 'Quantity': []}

            for pos in CAT_CARD_POSITIONS:
                cat_card_idx = src_i[pos]
                cat_features = self.wcst_env.get_card_features(cat_card_idx)
                
                if cat_features[0] == example_features[0]: # Colour match
                    match_indices['Colour'].append(pos.item())
                if cat_features[1] == example_features[1]: # Shape match
                    match_indices['Shape'].append(pos.item())
                if cat_features[2] == example_features[2]: # Quantity match
                    match_indices['Quantity'].append(pos.item())
            
            match_positions_batch.append(match_indices)
        return match_positions_batch

    def analyze_attention(self):
        head_feature_scores = {
            (layer, head): {'Colour': 0.0, 'Shape': 0.0, 'Quantity': 0.0, 'Count': 0}
            for layer in range(self.num_layers)
            for head in range(self.num_heads)
        }
        
        with torch.no_grad():
            for src, tgt in self.test_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]

                _, attn_maps_all = self.model(src, tgt_input)

                batch_match_positions = self._get_match_positions(src)
                
                for layer_idx, attn_maps in enumerate(attn_maps_all):
                    
                    cross_w = attn_maps["cross"] 

                    cross_attention_final_query = cross_w[:, :, -1, :] 
                    
                    for batch_idx in range(src.size(0)):
                        attn_batch = cross_attention_final_query[batch_idx] # (H, T_src)
                        match_pos = batch_match_positions[batch_idx]
                        
                        for feature_key in self.features:
                            positions = match_pos[feature_key]
                            
                            if not positions:
                                continue 

                            attn_sum = attn_batch[:, positions].sum(dim=-1) 
                            
                            for head_idx in range(self.num_heads):
                                key = (layer_idx, head_idx)
                                head_feature_scores[key][feature_key] += attn_sum[head_idx].item()
                                head_feature_scores[key]['Count'] += 1

        average_scores = {}
        for key, data in head_feature_scores.items():
            if data['Count'] > 0:
                average_scores[key] = {
                    f: data[f] / data['Count']
                    for f in self.features
                }
            else:
                average_scores[key] = {f: 0.0 for f in self.features}
                
        return average_scores

    def visualize_specialization(self, average_scores):

        
        all_data = []
        for (layer, head), scores in average_scores.items():
            for feature, score in scores.items():
                all_data.append({
                    'Layer': f'L{layer}', 
                    'Head': head, 
                    'Feature': feature, 
                    'Attention_Score': score
                })
                
        df = pd.DataFrame(all_data)

        fig, axes = plt.subplots(self.num_layers, 1, figsize=(10, 3 * self.num_layers), sharex=True)
        
        if self.num_layers == 1:
            axes = [axes] 

        for layer_idx, ax in enumerate(axes):
            layer_df = df[df['Layer'] == f'L{layer_idx}']
            pivot_table = layer_df.pivot_table(
                index='Head', 
                columns='Feature', 
                values='Attention_Score'
            )
            
            sns.heatmap(
                pivot_table, 
                annot=True, 
                fmt=".3f", 
                cmap="YlGnBu", 
                cbar_kws={'label': 'Average Attention Score'},
                ax=ax
            )
            ax.set_title(f'Decoder Cross-Attention Specialization - Layer {layer_idx}')
            ax.set_ylabel('Head Index')
            
        plt.tight_layout()
        plt.savefig('all_attention.png')
        plt.show()


if __name__ == '__main__':
    trainer.model.load_state_dict(torch.load('best_transformer_wcst.pth'))
    wcst_env = WCST(classification_batch) # Need an instance to use get_card_features
    
    # Replace these placeholders with your actual loaded objects
    # For demonstration, we'll assume a previously trained model is loaded
    
    # Recreate the WCST environment for testing purposes (needed for _get_match_positions)
    test_wcst_env = WCST(classification_batch) 
    
    # Assume the model and test_data were loaded and processed as in main file
    # For a real run, ensure you have a model trained on a fixed rule!
    
    # Re-instantiate the DataLoader (or just use the existing one from the Trainer)
    data_loader_instance = Dataset_Loader(
        training_batch=100000, 
        classification_batch=1000,
        train_split=0.0, val_split=0.0, test_split=1.0, 
        context_switch_interval=1000000
    )
    _, _, interpret_test_data = data_loader_instance.load_data()
    
    # Initialize the detector
    detector = RuleDetector(
        model=model, 
        test_data=interpret_test_data, 
        wcst_env=test_wcst_env, 
        batch_size=32
    )

    print("Analyzing attention head specialization...")
    avg_scores = detector.analyze_attention()
    
    print("\nFeature Attention Scores per Head (Average Attention to Matching Card Positions):")
    for (layer, head), scores in avg_scores.items():
        # Print the score and identify the specialized feature
        specialized_feature = max(scores, key=scores.get)
        print(f"L{layer} H{head}: C:{scores['Colour']:.4f} | S:{scores['Shape']:.4f} | Q:{scores['Quantity']:.4f} -> Specialized: {specialized_feature}")
        
    print("\nGenerating heatmap visualization...")
    detector.visualize_specialization(avg_scores)