from PIL import Image
import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor
from typing import Optional, Tuple
import numpy as np
from typing import Type
import einops
import argparse

torch_device = ("cuda" if torch.cuda.is_available() 
          else "mps" if torch.mps.is_available() 
          else  "cpu")
torch_dtype = (torch.bfloat16 if torch_device == "cuda" else torch.float32)

# -----------------------------------------------------------------------------
# Sequence alignment and Mapper Functions
# -----------------------------------------------------------------------------

### Replacement Task
def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Splits the text into words. If 'word_place' is a string, it finds all occurrences of the word in the text and stores their indices. 
    If 'word_place' is an integer, it wraps it in a list for consistent processing. 
    Encodes the text into tokens and decodes each token back into string form to identify the boundaries of each word in the tokenized version. 
    It iterates over these tokens, matching them to the specified word indices ('word_place') and collecting the corresponding token indices in the output list 'out'.
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    #print(f"word_place={word_place}, decoded_tokens={words_encode}")
    print(out)
    return np.array(out)

# def get_replacement_mapper_(x: str, y: str, tokenizer):
#     """
#     Splits both input strings x and y into words and constructs a mapping matrix of size max_len x max_len. 
#     """
#     """
#     Constructs a mapping matrix [max_len x max_len] that defines how attention should be
#     redirected from tokens in x to tokens in y. Handles only 1-to-1 and n-to-n mappings.
#     """
#     max_len = tokenizer.model_max_length  # usually 77
#     words_x = x.split()
#     words_y = y.split()
#     #mapper = np.identity((max_len), dtype=np.float32)
#     mapper = np.zeros((max_len, max_len), dtype=np.float32)
#     mapper[0][0] = 1.0
#     last_mapped_row = -1
#     last_mapped_col = -1

#     for i in range(min(len(words_x), len(words_y))):
#         inds_x = get_word_inds(x, i, tokenizer)
#         inds_y = get_word_inds(y, i, tokenizer)

#         # Skip special tokens like BOS/CLS
#         inds_x = [ix for ix in inds_x if ix > 0]
#         inds_y = [iy for iy in inds_y if iy > 0]

#         n = len(inds_x)
#         m = len(inds_y)

#         # Handle only 1-to-1 or n-to-n
#         # if n == m and n > 0:
#         #     for j in range(n):
#         #         mapper[inds_x[j]][inds_y[j]] = 1.0

        

#         if n == 1:
#             weights = m
#             for iy in inds_y:
#                 mapper[inds_x[0]][iy] = 1.0 / weights
#                 last_mapped_row = max(last_mapped_row, inds_x[0])
#                 last_mapped_col = max(last_mapped_col, iy)

              
#         # N-to-1 mapping: multiple tokens in x map to one in y
#         elif m == 1:
#             weights = n
#             for ix in inds_x:
#                 mapper[ix][inds_y[0]] = 1.0 / weights
#                 last_mapped_row = max(last_mapped_row, ix)
#                 last_mapped_col = max(last_mapped_col, inds_y[0])



#         start_row = last_mapped_row + 1
#         start_col = last_mapped_col + 1

#         for i in range(start_row, max_len):
#             if i < len(words_x) and words_x[i] == "<pad>":
#                 mapper[i][i] = 1.0
#             if i < len(words_y) and words_y[i] == "<pad>":
#                 mapper[i][i] = 1.0


#         print(mapper[0:15, 0:15])

#     return torch.from_numpy(mapper).to(torch_device, dtype=torch_dtype)

def get_replacement_mapper_(x: str, y: str, tokenizer):
    """
    Constructs a mapping matrix [max_len x max_len] that defines how attention 
    should be redirected from tokens in x to tokens in y. 

    Handles:
      - 1-to-1
      - 1-to-N
      - N-to-1
      - n-to-n

    Also sets diagonal to 1.0 for <pad> tokens after the main mappings.
    """
    max_len = tokenizer.model_max_length  # usually 77
    words_x = x.split()
    words_y = y.split()

    # Initialize the mapper matrix with zeros
    mapper = np.zeros((max_len, max_len), dtype=np.float32)

    # As an example, you might want the first token (e.g., <BOS> or <CLS>) to map to itself
    mapper[0][0] = 1.0  

    # Track the highest row/column used for mappings
    last_mapped_row = -1
    last_mapped_col = -1

    for i in range(min(len(words_x), len(words_y))):
        inds_x = get_word_inds(x, i, tokenizer)
        inds_y = get_word_inds(y, i, tokenizer)

        # Skip special tokens like BOS/CLS by removing index 0
        inds_x = [ix for ix in inds_x if ix > 0]
        inds_y = [iy for iy in inds_y if iy > 0]

        n = len(inds_x)  # number of tokens in x for word i
        m = len(inds_y)  # number of tokens in y for word i

        # --- 1) n == m (n-to-n) or 1-to-1 ---
        if n == m and n > 0:
            for j in range(n):
                mapper[inds_x[j]][inds_y[j]] = 1.0
            # Update last mapped row/col
            last_mapped_row = max(last_mapped_row, max(inds_x))
            last_mapped_col = max(last_mapped_col, max(inds_y))

        # --- 2) 1-to-N mapping ---
        elif n == 1 and m > 0:
            weight = 1.0 / m
            for iy in inds_y:
                mapper[inds_x[0]][iy] = weight
            # Update last mapped row/col
            last_mapped_row = max(last_mapped_row, inds_x[0])
            last_mapped_col = max(last_mapped_col, max(inds_y))

        # --- 3) N-to-1 mapping ---
        elif m == 1 and n > 0:
            weight = 1.0 / n
            for ix in inds_x:
                mapper[ix][inds_y[0]] = weight
            # Update last mapped row/col
            last_mapped_row = max(last_mapped_row, max(inds_x))
            last_mapped_col = max(last_mapped_col, inds_y[0])
        
        # 4) N-to-M (general case, both > 1)
        else:
            # Each cross pair of sub-tokens gets 1/(n*m)
            weight = 1.0 / max(n,m)
            for ix_ in inds_x:
                for iy_ in inds_y:
                    mapper[ix_][iy_] = weight
            last_mapped_row = max(last_mapped_row, max(inds_x))
            last_mapped_col = max(last_mapped_col, max(inds_y))

        # If neither condition is met, you can decide whether to skip, 
        # raise an error, or handle more complex alignments.

    # After finishing all mappings, fill diagonal for <pad> tokens
    start_row = last_mapped_row + 1
    start_col = last_mapped_col + 1
    num_pad = max_len - max(start_row, start_col)
    for j in range(num_pad):
        mapper[start_row + j][start_col + j] = 1.0
    # Convert to torch tensor if needed
    mapper = torch.from_numpy(mapper).to(torch_device, dtype=torch_dtype)
    return mapper


def get_replacement_mapper(prompts, tokenizer):
    """
    Given a list of prompts (with the first as the base), returns a stacked PyTorch tensor
    containing the mapping matrices for each modified prompt relative to the base.
    Each mapping matrix is of shape [max_len, max_len], where max_len is typically 77.
    """
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer)
        mappers.append(mapper)
    return torch.stack(mappers)

def get_replacement_mapper(prompts, tokenizer):
    """
    Given a list of prompts (with the first as the base), returns a stacked PyTorch tensor
    containing the mapping matrices for each modified prompt relative to the base.
    Each mapping matrix is of shape [max_len, max_len], where max_len is typically 77.
    """
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer)
        mappers.append(mapper)
    return torch.stack(mappers)



def get_replacement_mapper(prompts, tokenizer):
    """
    Returns stacked PyTorch tensor containing all the mapping matrices, where each matrix 
    corresponds to the mapping from the first prompt to one of the subsequent prompts.
    The max_len=77 because that is the maximum length of the CLIP text encoder.
    """
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer)
        mappers.append(mapper)
    return torch.stack(mappers)


# -----------------------------------------------------------------------------
# Image Processing Functions
# -----------------------------------------------------------------------------

def create_image_grid(images, rows=1, padding=10, bg_color="white"):
    """
    Creates a grid of images with the specified number of rows and padding.

    Args:
        images (list): List of PIL Image objects.
        rows (int): Number of rows for the grid.
        padding (int): Amount of padding (in pixels) between images.
        bg_color (str or tuple): Background color of the grid.

    Returns:
        PIL.Image: A single image containing the grid of input images.
    """
    # Calculate the number of columns needed
    num_images = len(images)
    cols = (num_images + rows - 1) // rows  # Ceiling division

    # Get the size of the largest image in the list
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    # Calculate the size of the grid canvas, including padding
    grid_width = cols * max_width + (cols + 1) * padding
    grid_height = rows * max_height + (rows + 1) * padding

    # Create a new blank canvas with the specified background color
    grid_image = Image.new("RGB", (grid_width, grid_height), bg_color)

    # Paste each image into its correct position in the grid with padding
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * max_width + (col + 1) * padding
        y = row * max_height + (row + 1) * padding
        grid_image.paste(img, (x, y))

    return grid_image

# -----------------------------------------------------------------------------
# Recursive Module Manipulation
# -----------------------------------------------------------------------------

def apply_to_all_named_modules(module: nn.Module, fn, parent_name:str=""):
    '''Recursively applies a function to all named modules in a PyTorch module.'''
    # Recurse through children with their instance names
    for name, child in module.named_children():
        # Construct the full name path for the current module
        full_name = parent_name + ("." if parent_name else "") + name
        # Apply the function to the current module
        fn(full_name, module, name, child)
        # Recurse into the child module
        apply_to_all_named_modules(child, fn, full_name)

def print_model_layers(model: nn.Module):
    '''Recursively prints the variable names of all layers in a PyTorch model and their type.'''
    apply_to_all_named_modules(
        model,
        lambda full_name, module, name, child: print(f"{full_name}: {child.__class__.__name__}")
    )
    
def replace_module_by_class_and_name(module: Type[nn.Module], target_class: str, target_name: str, replacement_class: Type[nn.Module], other_init_args: Tuple = ()):
    '''Recursively replaces all instances of `target_class` with `replacement_class` 
    in a PyTorch module.'''
    # Lambda function used to replace the target module with the replacement module
    def replace_module_by_class_and_name_fn(full_name, module, name, child):
        print(f"{full_name}: {child.__class__.__name__}")
        # If the current module is of the target class, replace it
        if name == target_name and child.__class__.__name__ == target_class:
            print("Replacing: ", target_class, replacement_class)
            setattr(module, name, replacement_class(child, *other_init_args))
    
    # Recursively apply the replacement function to all named modules
    apply_to_all_named_modules(
        module,
        replace_module_by_class_and_name_fn,
    )

def unet_inject_attention_modules(unet: UNet2DConditionModel, swapper: 'MySharedAttentionSwapper'): 
    # Which of these are the Self Attention Layers, and which are the Cross Attention Layers?
    replace_module_by_class_and_name(unet, 'Attention', "attn1", MyCrossAttention, [swapper])
    replace_module_by_class_and_name(unet, 'MyCrossAttention', "attn1", MyCrossAttention, [swapper])
    replace_module_by_class_and_name(unet, 'Attention', "attn2", MyCrossAttention, [swapper])
    replace_module_by_class_and_name(unet, 'MyCrossAttention', "attn2", MyCrossAttention, [swapper])

# -----------------------------------------------------------------------------
# Cross Attention Layer
# -----------------------------------------------------------------------------

class MySharedAttentionSwapper():
    
    def __init__(self, prompts, tokenizer, prop_steps_cross: float, prop_steps_self: float):
        # Initialize the global counters that are updated via the note_* methods
        self.cur_step = None # the current time step in diffusion
        self.cur_att_layer = None # the current attention layer (int) in UNet
        
        # Proportion of steps after which to replace cross/self attention
        self.prop_steps_cross = prop_steps_cross 
        self.prop_steps_self = prop_steps_self 
        # Mapper matrix for attention swapping
        self.mapper = get_replacement_mapper(prompts, tokenizer).to(torch_device)
        # Number of prompts
        self.num_prompts = len(prompts)
        
    def note_begin_diffusion(self, num_inference_steps: int):
        '''Take note of the beginning of the outer loop of diffusion.'''
        self.cur_step = -1
        self.cur_att_layer = -1
        
        # Update the number of steps for attention swapping
        self.num_steps_cross = int(num_inference_steps * self.prop_steps_cross)
        self.num_steps_self = int(num_inference_steps * self.prop_steps_self)
        
    def note_begin_diffusion_step(self, t):
        '''Take note of the beginning of a diffusion step.'''
        self.cur_step += 1
        self.cur_att_layer = 0
    
    def note_end_of_attention_layer(self):
        '''Take note of the end of an attention layer in UNet.'''
        self.cur_att_layer += 1
        
    def swap_attention_probs(self, attn_probs: torch.Tensor, is_cross_attn: bool) -> torch.Tensor:
        '''Swap attention probabilities based on the current state of the model.'''
        # We assume the first element of the batch corresponds to the original prompt.
        # Each other element in the batch corresponds to a prompt that may need attention swapping.
        if ((is_cross_attn and self.cur_step < self.num_steps_cross) or 
            (not is_cross_attn and self.cur_step < self.num_steps_self)):
            attn_base, attn_replace = attn_probs[0], attn_probs[1:]
            if is_cross_attn:
                ####TODO####
                
                # Swap attention probabilities for cross attention
                # if attn_replace.shape[-1] <= self.mapper.shape[-1]:
                #     attn_probs[1:] = torch.einsum("bhtj,bjk->bhtk",
                #                                   attn_replace,
                #
                # print("attn_base", attn_base.shape)
                # print("mapper",self.mapper.shape)
                # print("attn_replace",attn_replace.shape)
                
                # #self.mapper[:, :attn_replace.shape[-1], :attn_replace.shape[-1]]\
                # mapper = self.mapper.to(attn_base.dtype)
                # attn_probs[1:] = torch.einsum(
                #     "bhj,ajk->abhk",
                #     attn_base,  # replicate base for each edited prompt
                #     mapper  # trim mapper
                # )
                # Cast both attn_base and self.mapper to float32 before einsum.
                # Cast both tensors to float32

                # attn_base_f32 = attn_base.to(torch.float32)
                # mapper_f32 = self.mapper.to(torch.float32)
                # # Perform the einsum in float32
                # result_f32 = torch.einsum("bhj,ajk->abhk", attn_base_f32, mapper_f32)
                # # Cast the result back to attn_base's dtype (BFloat16)
                # attn_probs[1:] = result_f32.to(attn_base.dtype)

                # Replicate attn_base to add the batch dimension:
                # attn_base_rep = attn_base.unsqueeze(0).expand(self.mapper.shape[0], -1, -1, -1)

                # # Convert to float32 if necessary:
                # attn_base_f32 = attn_base_rep.to(torch.float32)
                # mapper_f32 = self.mapper.to(torch.float32)

                # # Perform the einsum operation in float32:
                # result_f32 = torch.einsum("bhtj,bjk->bhtk", attn_base_f32, mapper_f32)

                # # Convert the result back to the original dtype:
                # attn_probs[1:] = result_f32.to(attn_base.dtype)
                attn_probs[1:] = torch.matmul(attn_base.unsqueeze(0), self.mapper.unsqueeze(1))


                ####END_TODO####
            else:
                if attn_replace.shape[2] <= 16 ** 2:
                    attn_probs[1:] = attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
                else:
                    attn_probs[1:] = attn_replace
        return attn_probs
        
class MyCrossAttention(nn.Module):
    
    def __init__(self, attn: nn.Module, swapper: MySharedAttentionSwapper):
        super().__init__()
        self.swapper = swapper
        if attn.__class__.__name__ == 'MyCrossAttention':
          # Grab the inner Attention class
          attn = attn.attn
        self.attn = attn
        assert attn.spatial_norm is None
        assert attn.norm_cross is None
        assert attn.residual_connection == False
        assert attn.rescale_output_factor == 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # For Prompt to Prompt we need to know whether or not we are in cross or self attention
        is_cross_attn = True if encoder_hidden_states is not None else False
        
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = self.attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.attn.norm_cross:
            encoder_hidden_states = self.attn.norm_encoder_hidden_states(encoder_hidden_states)

        ####TODO####
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(encoder_hidden_states)
        value = self.attn.to_v(encoder_hidden_states)

        query = self.attn.head_to_batch_dim(query)
        key = self.attn.head_to_batch_dim(key)
        value = self.attn.head_to_batch_dim(value)

        attention_probs = self.attn.get_attention_scores(query, key, attention_mask)
        ####END_TODO####

        # -----------------------------------------------------------------------------
        # Prompt-to-prompt attention swapping
        # -----------------------------------------------------------------------------
        # For prompt to prompt, we may need to swap in a new set of attention probabilities
        # depending on the current state of the model.
        atp_shape_before = attention_probs.shape
        # Only modify the attention probabilities for the conditional inputs
        attention_probs_cond = attention_probs[attention_probs.shape[0]//2:]
        # Reshape to: [batch_size, num_heads, seq_len, seq_len]
        attention_probs_cond = einops.rearrange(attention_probs_cond, '(b h) t s -> b h t s', b=self.swapper.num_prompts) 
        attention_probs_cond = self.swapper.swap_attention_probs(attention_probs_cond, is_cross_attn)
        # Reshape to: [batch_size * num_heads, seq_len, seq_len]
        attention_probs_cond = einops.rearrange(attention_probs_cond, 'b h t s -> (b h) t s')
        attention_probs[attention_probs.shape[0]//2:] = attention_probs_cond

        atp_shape_after = attention_probs.shape
        assert atp_shape_before == atp_shape_after
        # -----------------------------------------------------------------------------
        
        ####TODO####
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.attn.batch_to_head_dim(hidden_states)
        hidden_states = self.attn.to_out[0](hidden_states)
        hidden_states = self.attn.to_out[1](hidden_states)  # dropout
        hidden_states = (hidden_states / self.attn.rescale_output_factor)
        #hidden_states = hidden_states + residual
        ####END_TODO####

        self.swapper.note_end_of_attention_layer()
        return hidden_states
    
# -----------------------------------------------------------------------------
# Latent Diffusion Model Pipeline
# -----------------------------------------------------------------------------


class MyLDMPipeline():
    '''
    Latent diffusion model pipeline for generating images from text prompts.
    '''
    
    def __init__(self, num_inference_steps, guidance_scale):
        # Load pretrained models.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch_dtype, use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", torch_dtype=torch_dtype,)
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", torch_dtype=torch_dtype, use_safetensors=True)
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch_dtype, use_safetensors=True)
        self.scheduler = PNDMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler", torch_dtype=torch_dtype)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="feature_extractor", torch_dtype=torch_dtype)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="safety_checker", torch_dtype=torch_dtype, use_safetensors=True)

        # Move models to selected device
        self.vae.to(torch_device)
        self.text_encoder.to(torch_device)
        self.unet.to(torch_device)
        self.safety_checker.to(torch_device)

        self.num_inference_steps = num_inference_steps  # Number of denoising steps
        self.guidance_scale = guidance_scale  # Scale for classifier-free guidance

    @staticmethod
    def get_random_noise(batch_size: int, channel: int, height: int, width: int, generator: torch.Generator, same_noise_in_batch=True) -> torch.Tensor:
        '''Generate random noise of the specified shape.'''
        
        if same_noise_in_batch:
            ####TODO####
            # use the same noise for all entries in the batch
            # Your code here
            latent = torch.randn(
                (1, channel, height, width),
                generator=generator,
                device=torch_device,
                dtype=torch_dtype,
            )
            ####END_TODO####
            latents = latent.expand(batch_size, channel, height, width)
        else:
            # use different noise for each entry in the batch
            latents = torch.randn(
                (batch_size, channel, height, width),
                generator=generator,
                device=torch_device,
                dtype=torch_dtype,
            )
        return latents
        


    def generate_image_from_text(self, prompt: list, swapper: MySharedAttentionSwapper) -> Image:
        return MyLDMPipeline._generate_image_from_text(prompt, self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler, self.feature_extractor, self.safety_checker, swapper, self.num_inference_steps, self.guidance_scale)
        
    @staticmethod
    def _generate_image_from_text(prompt: list, vae: AutoencoderKL, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, unet: UNet2DConditionModel, scheduler: PNDMScheduler, feature_extractor: CLIPImageProcessor, safety_checker: StableDiffusionSafetyChecker, swapper: MySharedAttentionSwapper,
                                  num_inference_steps: int, guidance_scale: float, same_noise_in_batch=True) -> Image:
        # 0. Set default parameters
        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        generator = torch.Generator(device=torch_device)  # Move generator to correct device
        generator.manual_seed(1024)  # Seed the generator
        batch_size = len(prompt)

        # 1. Tokenize and embed the text
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 2. Create random noise
        latents = MyLDMPipeline.get_random_noise(batch_size, unet.config.in_channels, height // 8, width // 8, generator, same_noise_in_batch)
        
        # 3. Denoise the image
        latents = latents * scheduler.init_noise_sigma
        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        swapper.note_begin_diffusion(num_inference_steps)
        for t in tqdm(scheduler.timesteps):
            swapper.note_begin_diffusion_step(t)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 4. Decode the image
        
        # scale and decode the image latents with vae
        latents = 1.0 / vae.config.scaling_factor * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        image = image_processor.postprocess(image, output_type="pil")

        return image
       