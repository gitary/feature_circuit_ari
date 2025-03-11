import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import nnsight
from nnsight import NNsight
from nnsight import LanguageModel

from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

class EmbeddingAnalysis:
    def __init__(self, llm, text):
        self.llm = llm
        self.text = text
        self.input_embed = None
        self.cosine_similarity_matrix = None
        self.euclidean_distance_matrix = None
        self.lst_embeddings_through_blocks = None
        self.lst_cos_through_blocks = []
        self.lst_euc_through_blocks = []
        self.lst_L2_norm_through_blocks = []

    def trace_embeddings(self):
        with self.llm.trace(self.text):
            if isinstance(self.llm._model, GPT2LMHeadModel):
                self.input_embed = self.llm.transformer.drop.input.save()                     

            if isinstance(self.llm._model, Gemma2ForCausalLM):
                self.input_embed = self.llm.model.embed_tokens.output.save()
            
            else: 
                print("Model not supported")

    def trace_embeddings_through_blocks(self):
        #residual_after_mlp = gemma.model.layers[2].mlp.output.save()
        self.trace_embeddings()        
        self.lst_embeddings_through_blocks = [self.input_embed]
        with self.llm.trace(self.text):
            if isinstance(self.llm._model, GPT2LMHeadModel):
                for i in range(len(self.llm.transformer.h)):
                    residual_after_mlp = self.llm.transformer.h[i].mlp.output.save()
                    self.lst_embeddings_through_blocks.append(residual_after_mlp)

            if isinstance(self.llm._model, Gemma2ForCausalLM):
                for i in range(len(self.llm.model.layers)):
                    residual_after_mlp = self.llm.model.layers[i].mlp.output.save()
                    self.lst_embeddings_through_blocks.append(residual_after_mlp)    
            
            else: 
                print("Model not supported")

        return self.lst_embeddings_through_blocks

    def calculate_cosine_similarity_matrix(self, embed=None):
        if embed is None:
            embed = self.input_embed
        input_embed_normalization = F.normalize(embed[0], p=2, dim=1)  # L2 normalization
        self.cosine_similarity_matrix = torch.mm(input_embed_normalization, input_embed_normalization.t())

    def calculate_euclidean_distance_matrix(self, embed=None):
        if embed is None:
            embed = self.input_embed
        self.euclidean_distance_matrix = torch.cdist(embed[0], embed[0], p=2)

    def calculate_L2_norm(self, embed=None):
        if embed is None:
            embed = self.input_embed
        self.input_embed_norm = torch.norm(embed[0], p=2, dim=1)  # Calculate L2 norm

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

        # Define common plot parameters
        fig_width = 7
        fig_height = 2
        x_range = range(len(self.lst_cos_through_blocks))

        # Plotting function
        def plot_metric_through_blocks(metric_list, ylabel, title, label):
            plt.figure(figsize=(fig_width, fig_height))
            plt.plot(x_range, metric_list, 'o--', label=label)
            plt.xlabel('Block Index')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.show()

        # Plotting the metrics through blocks
        plot_metric_through_blocks(self.lst_cos_through_blocks, 'Mean Cosine Similarity', 'Mean Cosine Similarity Through Blocks', 'Mean Cosine Similarity')
        plot_metric_through_blocks(self.lst_euc_through_blocks, 'Mean Euclidean Distance', 'Mean Euclidean Distance Through Blocks', 'Mean Euclidean Distance')
        plot_metric_through_blocks(self.lst_L2_norm_through_blocks, 'Mean L2 norm', 'Mean embedding\'s L2 norm Through Blocks', 'Mean embedding\'s L2 norm')
