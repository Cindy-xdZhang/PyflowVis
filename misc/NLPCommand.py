# from GuiObjcts.Object import Object
# import logging
# from datasets import load_dataset
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import Trainer, TrainingArguments
# import imgui

# def finetue_nlp():
#     dataset = load_dataset("emotion")
#     model_name = "bert-base-uncased"
#     num_labels = 6
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     def encode(examples):
#         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
#     encoded_dataset = dataset.map(encode, batched=True)
#     encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

#     training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     evaluation_strategy="epoch",  # 每个epoch进行一次评估
#     )

#     trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     )

#     trainer.train()
#     trainer.evaluate()







# class NLPCommandObject(Object):
#     def __init__(self):
#         super().__init__("NLPCommand")
#         self.textBufferLengthMax = 512
#         self.queryText=""
#         self.answer = None
#         self.mode=          "MyModel"  
#         self.addAction("Query", lambda obj: obj.Query())
#         #["OpenAI","MyModel"]
#         # self.addAction("Clear", lambda obj: obj.clear())
#         # self.init_GPt()

#     # def init_GPt(self):
#     #     mykey="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"
#     #     self.client = OpenAI(api_key= mykey)
#     #     self.queryText=""
#     #     self.answer = None
#     # def QueryOpenAi(self,question:str):
#     #     try:
#     #         completion = self.client.chat.completions.create(
#     #         model="gpt-3.5-turbo",
#     #         messages=[
#     #             {"role": "system", "content": "You are a flow visualization program assistant, skilled in explaining complex programming concepts with flow visualization."},
#     #             {"role": "user", "content": question}
#     #         ]
#     #         )
#     #         self.answer =  str(completion.choices[0].message) 
#     #     except Exception  as e:
#     #         logging.error(f"Query OpenAI with an error occurred: {e}")

#     def Query(self,q:str=None):
#         if self.mode=="MyModel":
#             self.QueryMymodel(q)


#     def QueryMymodel(self,q:str=None):
#         try:
#             self.answer =  "This is MyModel's answer to:"+ q

#         except Exception  as e:
#             logging.error(f"Query MyModel with an error occurred: {e}")
        



#     def  drawGui(self):
#         if self.GuiVisible:
#             _,self.GuiVisible=imgui.begin(self.name,self.GuiVisible)
#             cgd,self.queryText=imgui.input_text("QueryText", self.queryText, self.textBufferLengthMax)
#             if imgui.button("Query"):
#                 self.Query(self.queryText)
#             if self.answer is not None:
#                 imgui.text(self.answer)
            
#             imgui.end()
#         return 
    
   

# if __name__ == "__main__":
#     finetue_nlp()