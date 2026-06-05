import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from loguru import logger

class AMRBARTDecoderWrapper(nn.Module):
    def __init__(self, model_name: str = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing", gat_out_channels: int = 256):
        """
        Wrapper around the pretrained AMRBART sequence-to-sequence model
        to support decoding directly from projected graph embeddings.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Hugging Face AMRBART
        logger.info(f"Loading AMRBART model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Freezing the weights of BART to focus training on the graph encoder,
        # or keeping it fine-tuneable depending on resource availability.
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Linear projection to map GAT output channels (e.g. 256)
        # to BART-large's encoder hidden state dimension (1024)
        self.bart_hidden_dim = self.model.config.d_model  # 1024 for BART-large
        self.emb_projection = nn.Linear(gat_out_channels, self.bart_hidden_dim).to(self.device)
        logger.info(f"BART hidden dim: {self.bart_hidden_dim}. Created projection from {gat_out_channels} to {self.bart_hidden_dim}.")

    def generate_amr(self, projected_emb: torch.Tensor, max_length: int = 256, num_beams: int = 5) -> list[str]:
        """
        Args:
            projected_emb: (batch_size, sequence_length, gat_out_channels) tensor
                           representing the nodes of the graph projected to the AMR space.
        Returns:
            decoded_amrs: List of generated AMR graph strings.
        """
        self.model.eval()
        self.emb_projection.eval()
        
        batch_size = projected_emb.size(0)
        device = projected_emb.device
        
        with torch.no_grad():
            # 1. Project embedding to BART's input dimension
            # (batch_size, seq_len, 256) -> (batch_size, seq_len, 1024)
            bart_inputs = self.emb_projection(projected_emb.to(self.device))
            
            # 2. Package into BaseModelOutput as expected by Hugging Face generate
            encoder_outputs = BaseModelOutput(last_hidden_state=bart_inputs)
            
            # 3. Create attention mask (all 1s for the sequence length)
            seq_len = projected_emb.size(1)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
            
            # 4. Generate token IDs using the decoder
            generated_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            # 5. Decode back to AMR strings
            decoded_amrs = []
            for g_id in generated_ids:
                amr_str = self.tokenizer.decode(g_id, skip_special_tokens=True)
                decoded_amrs.append(amr_str)
                
        return decoded_amrs
