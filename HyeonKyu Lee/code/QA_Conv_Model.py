import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

class QAConvModel(nn.Module):
    def __init__(self, model_name, model_config):
        super().__init__() 
        self.model_name = model_name
        self.model_config = model_config
        self.backbone = AutoModel.from_pretrained(model_name, model_config)
        self.query_layer = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=True)
        self.query_calssify_layer = nn.Linear(model_config.hidden_size, 6, bias=True)
        self.key_layer = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=True)
        self.value_layer = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=True)
        self.conv1d_layer1 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=1)
        self.conv1d_layer3 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=5, padding=2)
        self.drop_out = nn.Dropout(0.7)
        self.classify_layer = nn.Linear(model_config.hidden_size*3, 2, bias=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None, question_type=None):
        if "xlm" in self.model_name:
            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        sequence_output = outputs[0] # (B * 384 * 1024)

        embedded_query = sequence_output * (token_type_ids==0) # 전체 Text 중 query에 해당하는 Embedded Vector만 남김.
        embedded_query = self.query_layer(embedded_query) # Dense Layer를 통과 시킴. (B * max_seq_length * hidden_size)
        embedded_query = torch.mean(embedded_query, 1, keepdim=True) # Query에 해당하는 Token Embedding을 평균냄. (B * 1 * hidden_size)
        query_logits = self.query_calssify_layer(embedded_query.squeeze(1)) # Query의 종류를 예측하는 Branch (B * 6)

        embedded_key = self.key_layer(sequence_output) # (B * max_seq_length * hidden_size)
        embedded_value = self.value_layer(sequence_output) # (B * max_seq_length * hidden_size)
        attention_rate = torch.matmul(embedded_key, torch.transpose(embedded_query, 1, 2)) # Context의 Value Vector와 Quetion의 Query Vector를 사용
        attention_rate = F.softmax(attention_rate, -1) # Question과 Context의 Attention Rate를 구함. (B * max_seq_legth)
        embedded_value = embedded_value * attention_rate # Attention Rate를 활용해서 Output 값을 변경함. (B * max_seq_legth * hidden_size)

        conv_input = embedded_value.transpose(1, 2) # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_output1 = F.relu(self.conv1d_layer1(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv1d_layer3(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv1d_layer5(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1) # Concatenation (B * num_conv_filter x 3 * max_seq_legth)

        concat_output = concat_output.transpose(1, 2) # Dense Layer에 입력을 위해 Transpose (B * max_seq_legth * num_conv_filter x 3)
        concat_output = self.drop_out(concat_output) # dropout 통과
        logits = self.classify_layer(concat_output) # Classifier Layer를 통해 최종 Logit을 얻음. (B * max_seq_legth * 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return {"start_logits" : start_logits, "end_logits" : end_logits, "hidden_states" :  outputs.hidden_states, "attentions" : outputs.attentions, "query_logits" : query_logits} 