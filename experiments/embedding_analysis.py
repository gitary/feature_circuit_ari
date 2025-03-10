# filepath: /home/ailab/Code/MechInt/feature-circuits/experiments/embedding_analysis.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import nnsight
from nnsight import NNsight
from nnsight import LanguageModel



class EmbeddingAnalysis:
    def __init__(self, llm, text):
        self.llm = llm
        self.text = text
        self.input_embed = None
        self.cosine_similarity_matrix = None
        self.euclidean_distance_matrix = None

    def trace_embeddings(self):
        with self.llm.trace(self.text):
            self.input_embed = self.llm.transformer.drop.input.save()

    def calculate_cosine_similarity_matrix(self):
        input_embed_normalization = F.normalize(self.input_embed[0], p=2, dim=1)  # L2 normalization
        self.cosine_similarity_matrix = torch.mm(input_embed_normalization, input_embed_normalization.t())

    def calculate_euclidean_distance_matrix(self):
        self.euclidean_distance_matrix = torch.cdist(self.input_embed[0], self.input_embed[0], p=2)

    def plot_pairwise_matrix(self, pairwise_matrix, title):
        tokens = self.llm.tokenizer.encode(self.text)
        tokens_str = [self.llm.tokenizer.decode(tokens[i]) for i in range(len(tokens))]
        plt.figure(figsize=(10, 8))
        sns.heatmap(pairwise_matrix.cpu().detach().numpy(), annot=True, cmap='coolwarm', fmt=".2f", xticklabels=tokens_str, yticklabels=tokens_str)
        plt.title(title)
        plt.xlabel("Token")
        plt.ylabel("Token")
        plt.show()

    def analyze(self):
        self.trace_embeddings()
        self.calculate_cosine_similarity_matrix()
        self.calculate_euclidean_distance_matrix()        
        
        print(f"Mean Cosine Similarity: {torch.mean(self.cosine_similarity_matrix)}")
        self.plot_pairwise_matrix(self.cosine_similarity_matrix, "Pairwise Cosine Similarity Matrix")

        print(f"Mean Euclidean Distance: {torch.mean(self.euclidean_distance_matrix)}")
        self.plot_pairwise_matrix(self.euclidean_distance_matrix, "Pairwise Euclidean Distance Matrix")
        #print("Euclidean Distance Matrix:", self.euclidean_distance_matrix)
        #print("Euclidean Distance Matrix shape:", self.euclidean_distance_matrix.shape)
        #print("Cosine Similarity Matrix shape:", self.cosine_similarity_matrix.shape)        
    
    def get_mean_cosine_similarity(self):
        self.trace_embeddings()
        self.calculate_cosine_similarity_matrix()
        self.calculate_euclidean_distance_matrix()
        
        self.mean_cosine_similarity = torch.mean(self.cosine_similarity_matrix)
       
        return self.mean_cosine_similarity.item()
    
    def get_mean_euclidean_distance(self):
        self.trace_embeddings()
        self.calculate_cosine_similarity_matrix()
        self.calculate_euclidean_distance_matrix()
        
        self.mean_euclidean_distance = torch.mean(self.euclidean_distance_matrix)
        
        return self.mean_euclidean_distance.item()
    
    ######### Work in progress #########
    def trace_embeddings_through_blocks(self):
        with self.llm.trace(self.text):
            self.input_embed = self.llm.transformer.drop.input.save()
            # Access the residual stream after the MLP
            self.lst_embeddings_through_blocks = [self.input_embed]
            for i in range(len(self.llm.transformer.h)):
                residual_after_mlp = self.llm.transformer.h[i].mlp.output.save()
                self.lst_embeddings_through_blocks.append(residual_after_mlp)
        
        return self.lst_embeddings_through_blocks        

                
            
            
    def analyze_through_blocks(self):
        for embs in self.trace_embeddings_through_blocks(): 
            self.calculate_cosine_similarity_matrix()
            self.calculate_euclidean_distance_matrix()        
            
            print(f"Mean Cosine Similarity: {torch.mean(self.cosine_similarity_matrix)}")
            self.plot_pairwise_matrix(self.cosine_similarity_matrix, "Pairwise Cosine Similarity Matrix")

            print(f"Mean Euclidean Distance: {torch.mean(self.euclidean_distance_matrix)}")
            self.plot_pairwise_matrix(self.euclidean_distance_matrix, "Pairwise Euclidean Distance Matrix")
            #print("Euclidean Distance Matrix:", self.euclidean_distance_matrix)
            #print("Euclidean Distance Matrix shape:", self.euclidean_distance_matrix.shape)
            #print("Cosine Similarity Matrix shape:", self.cosine_similarity_matrix.shape)





class BlocksEmbeddingAnalysis:
    def __init__(self, llm, text):
        self.llm = llm
        self.text = text
        self.input_embed = None
        self.cosine_similarity_matrix = None
        self.euclidean_distance_matrix = None
    
    def trace_embeddings_through_blocks(self):
        with self.llm.trace(self.text):
            self.input_embed = self.llm.transformer.drop.input.save()
            # Access the residual stream after the MLP
            self.lst_embeddings_through_blocks = [self.input_embed]
            for i in range(len(self.llm.transformer.h)):
                residual_after_mlp = self.llm.transformer.h[i].mlp.output.save()
                self.lst_embeddings_through_blocks.append(residual_after_mlp)
        
        return self.lst_embeddings_through_blocks   


    def calculate_cosine_similarity_matrix(self, embed):
        input_embed_normalization = F.normalize(embed[0], p=2, dim=1)  # L2 normalization
        self.cosine_similarity_matrix = torch.mm(input_embed_normalization, input_embed_normalization.t())

    def calculate_L2_norm(self, embed):
        self.input_embed_norm = torch.norm(embed[0], p=2, dim=1)  # Calculate L2 norm      
       

    def calculate_euclidean_distance_matrix(self, embed):
        self.euclidean_distance_matrix = torch.cdist(embed[0], embed[0], p=2)


    def analyze_through_blocks(self):
        self.lst_cos_through_blocks = []
        self.lst_euc_through_blocks = []
        self.lst_L2_norm_through_blocks = []

        for emb in self.trace_embeddings_through_blocks(): 
            self.calculate_cosine_similarity_matrix(emb)
            self.calculate_euclidean_distance_matrix(emb)
            self.calculate_L2_norm(emb)        
            self.lst_cos_through_blocks.append(torch.mean(self.cosine_similarity_matrix).item())
            self.lst_euc_through_blocks.append(torch.mean(self.euclidean_distance_matrix).item())
            self.lst_L2_norm_through_blocks.append(torch.mean(self.input_embed_norm).item())       

        # Plotting the mean cosine similarity through blocks
        fig_width = 7
        fig_height = 2

        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(range(len(self.lst_cos_through_blocks)), self.lst_cos_through_blocks, 'o--', label='Mean Cosine Similarity')
        plt.xlabel('Block Index')
        plt.ylabel('Mean Cosine Similarity')
        plt.title('Mean Cosine Similarity Through Blocks')
        plt.legend()
        plt.show()

        # Plotting the mean Euclidean Distance through blocks
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(range(len(self.lst_euc_through_blocks)), self.lst_euc_through_blocks, 'o--', label='Mean Euclidean Distance')
        plt.xlabel('Block Index')
        plt.ylabel('Mean Euclidean Distance')
        plt.title('Mean Euclidean Distance Through Blocks')
        plt.legend()
        plt.show()

        # Plotting the mean L2 norm through blocks
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(range(len(self.lst_L2_norm_through_blocks)), self.lst_L2_norm_through_blocks, 'o--', label='Mean embedding\'s L2 norm')
        plt.xlabel('Block Index')
        plt.ylabel('Mean L2 norm')
        plt.title('Mean embedding\'s L2 norm Through Blocks')
        plt.legend()
        plt.show()

        #return self.lst_cos_through_blocks, self.lst_euc_through_blocks
        

    

                
            
            



