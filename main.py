from transformers import AutoTokenizer,AutoModel


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True).half().cuda()
prompt_text = "小孩牙龈中通服用什么药"
"--------------------------------------------------------------------------------------------------"
print("普通Chat GLM询问结果：")
response,_= model.chat(tokenizer,prompt_text,history=[])
print(response)