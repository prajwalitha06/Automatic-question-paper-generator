import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import askopenfile
import tkinter.messagebox
import os
import tensorflow as tf
import keras
from keras import applications
import numpy as np
import easygui
import os
import serial
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import json
import requests
import string
import re
import nltk
import itertools
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import textwrap
import fpdf
from PIL import Image,ImageTk
from pdf2image import convert_from_path
from tkPDFViewer import tkPDFViewer as pdfsd
import tkinter.messagebox
import textract
from pdf2image import convert_from_path
import webbrowser
from datetime import datetime
from textwrap3 import wrap
import torch
import random
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
import string
import traceback
from flashtext import KeywordProcessor
import fpdf
import spacy
nlp = spacy.load('en_core_web_sm')
import webbrowser
import pikepdf
from tqdm import tqdm
import sqlite3
import webbrowser
import subprocess
import PyPDF2








def question():
    my_w =Toplevel(root)
    my_w.geometry('1350x710+0+10')  
    my_w.title('Question Paper Generator')
    my_font1=('times', 18, 'bold')

    my_w.configure(bg='black')
        
    my_w.grab_set()

    '''bg = PhotoImage(file_path="backgroundpic.png")
    bgLabel = Label(my_w, image=bg)
    bgLabel.place(x=0, y=0)'''


    def qsngenerator():
        my_qsn=Toplevel(root)
        my_qsn.geometry('1350x710+0+10')  
        my_qsn.title('Question Paper Generator')
        my_font1=('times', 18, 'bold')


        my_qsn.configure(bg='black')
        
        my_qsn.grab_set()


        l1 = tk.Label(my_qsn,text='\n Upload paragraphed  Text file\n & \n get your Question paper \n',width=45,font=('italic',20,'bold'),bg='DarkTurquoise',
                        fg='black',)  
        l1.place(x=350, y=250,width=420)


        #-------------------- upload button ----------------------------------------------
        b1 = tk.Button(my_qsn, text='UPLOAD FILES', 
        width=20,command = lambda:qsnpaper(), activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='yellow')
        b1.place(x=80,y=600, width=180, height=40)
        #----------------------------------------------------------------------------------




        print(tf.__version__)
        print(".\n. \n. \n. \n. \n. \n. \n. \n. \n. \n. \n. \n")
        print("Question Paper Generator Is Loading.....")
        titleLabel = Label(my_qsn, text='Question Paper Generator ', font=('italic', 25, 'bold '), bg='black',fg='DarkTurquoise', )
        titleLabel.place(x=17, y=8,width=1100, height=80)

        def result():
            text =upload_file()
            for wrp in wrap(text, 150):
                print("")
            from transformers import T5ForConditionalGeneration,T5Tokenizer
            summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
            summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')



            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            summary_model = summary_model.to(device)




            
            
                    





            def set_seed(seed: int):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            set_seed(42)


            def postprocesstext (content):
                final=" "
                for sent in sent_tokenize(content):
                    sent = sent.capitalize()
                    final = final +" "+sent
                return final


            def summarizer(text,model,tokenizer):
                text = text.strip().replace("\n"," ")
                text = "summarize: "+text
                max_len = 512
                encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

                input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

                outs = model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                early_stopping=True,
                                                num_beams=3,
                                                num_return_sequences=1,
                                                no_repeat_ngram_size=2,
                                                min_length = 75,
                                                max_length=300)


                dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
                summary = dec[0]
                summary = postprocesstext(summary)
                summary= summary.strip()
                return summary
            summarized_text = summarizer(text,summary_model,summary_tokenizer)


            
            for wrp in wrap(text, 150):
                print("")
            print ("\n")
            for wrp in wrap(summarized_text, 500):
                print("")
            print ("\n")


            def get_nouns_multipartite(content):
                out=[]
                try:
                    extractor = pke.unsupervised.MultipartiteRank()
                    extractor.load_document(input=content,language='en')
                    pos = {'PROPN','NOUN'}
                    stoplist = list(string.punctuation)
                    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
                    stoplist += stopwords.words('english')
                    extractor.candidate_selection(pos=pos)
                    extractor.candidate_weighting(alpha=1.1,
                                                threshold=0.75,
                                                method='average')
                    keyphrases = extractor.get_n_best(n=15)
                    

                    for val in keyphrases:
                        out.append(val[0])
                except:
                    out = []
                    traceback.print_exc()

                return out

            def get_keywords(originaltext,summarytext):
                keywords = get_nouns_multipartite(originaltext)
                keyword_processor = KeywordProcessor()
                for keyword in keywords:
                    keyword_processor.add_keyword(keyword)

                keywords_found = keyword_processor.extract_keywords(summarytext)
                keywords_found = list(set(keywords_found))

                important_keywords =[]
                for keyword in keywords:
                    if keyword in keywords_found:
                        important_keywords.append(keyword)

                return important_keywords[:20]


            imp_keywords = get_keywords(text,summarized_text)
            question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
            question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
            question_model = question_model.to(device)

            def get_question(context,answer,model,tokenizer):
                text = "context: {} answer: {}".format(context,answer)
                encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
                input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

                outs = model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                early_stopping=True,
                                                num_beams=5,
                                                num_return_sequences=1,
                                                no_repeat_ngram_size=2,
                                                max_length=72)


                dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
                Question = dec[0].replace("question:","")
                Question= Question.strip()
                return Question

            for wrp in wrap(summarized_text, 500):
                print("")
            print ("\n")


            quesful=[]
            for answer in imp_keywords:
                ques = get_question(summarized_text,answer,question_model,question_tokenizer)
                print ("\n")
                quesful.append(ques)

            
            print("Question paper is Loading...!") 

            pdf = fpdf.FPDF(format='letter')
            pdf.add_page()
            pdf.set_font("Arial", size=30)
            pdf.write(30,"                    Question Paper")
            pdf.ln()
            pdf.set_font("Arial", size=12)


            for i in quesful:
                pdf.write(10,str(i))
                pdf.ln(15)
            pdf.output("questionpaperqsn.pdf")

            

            
            return pdf

             


        def qsnpaper():
            finalprint =result()
            
            l3 = tk.Label(my_qsn,text='\n Generated Question Paper are \n successfully saved as \n \n',font=('italic', 20, 'bold '),fg='white',bg="black")
            l3.place(x=350, y=250,width=420)
            l5 = tk.Label(my_qsn,text='questionpaperqsn.pdf',font=('italic', 20, 'bold '),
                                fg='DarkTurquoise',bg="black")  
            l5.place(x=395, y=350)
                
            #----------SEE PDF Button----------------------------------------------------------------------------------------

            b2 = tk.Button(my_qsn, text='SEE PDF', 
            width=20,command = lambda:see_pdf(), activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='DarkTurquoise')
            b2.place(x=470,y=600, width=155, height=40)

            #------------------------------------------------------------------------------------------------------------

            #----------CLOSE Button----------------------------------------------------------------------------------------

            b3 = tk.Button(my_qsn, text='CLOSE', 
            width=20,command = lambda:close(my_qsn), activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='yellow')
            b3.place(x=790,y=600, width=140, height=40)

            #------------------------------------------------------------------------------------------------------------
            return 0    


        def upload_file():
            l10 = tk.Label(my_qsn,text='\n Wait ! \n Questions are Generating \n',width=45,font=('italic',20,'bold'),bg='DarkTurquoise',
                        fg='black',)  
            l10.place(x=350, y=250,width=420)
            
            f_types = [('Text Files', '*.txt'),('Doc Files','*.docx')]   
            filename = tk.filedialog.askopenfilename(multiple=False,filetypes=f_types)
            if filename:
                try:
                    with open(filename, "r+", encoding='utf-8') as file:
                        text_content = file.read()
                except UnicodeDecodeError as error:
                    with open(filename, "r+", encoding='ISO-8859-1') as file:
                        text_content = file.read()
                return text_content
            else:
                messagebox.showwarning("Alert", "No file selected!")
                return None

        

        def see_pdf():

            
            def save_pdf_with_password(input_path, output_path, password):
                pdf = pikepdf.open(input_path, allow_overwriting_input=True)
                pdf.save(output_path, encryption=pikepdf.Encryption(user=password))
                pdf_location=r"C:/Users/prajwalitha\Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaper.pdf"
                webbrowser.open(pdf_location)
            
            input_pdf_path = "C:/Users/prajwalitha/Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaperqsn.pdf"
            output_pdf_path = "C:/Users/prajwalitha/Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaper.pdf"
            password = "rmkcet"
            save_pdf_with_password(input_pdf_path, output_pdf_path, password)

    def close(window):
        window.destroy()




    def fillupgen():
        my_fill=Toplevel(root)
        my_fill.geometry('1350x710+0+10')  
        my_fill.title('Fill in the Blanks Generator')
        my_font1=('times', 18, 'bold')

        my_fill.configure(bg='black')
        

        my_fill.grab_set()

        l1 = tk.Label(my_fill,text='\n Upload paragraphed  Text file\n & \n get Fill up Question paper \n',width=45,font=('italic',20,'bold'),bg='DarkTurquoise',
                        fg='black',)  
        l1.place(x=350, y=250,width=420)


        #-------------------- upload button ----------------------------------------------
        b1 = tk.Button(my_fill, text='UPLOAD FILES', 
        width=20,command = lambda:qsnpaper(), activebackground='skyblue',font=('italic',17,'bold'),bg='black',fg='yellow')
        b1.place(x=80,y=600, width=180, height=40)
        #----------------------------------------------------------------------------------




        print(tf.__version__)
        print(".\n. \n. \n. \n. \n. \n. \n. \n. \n. \n. \n. \n")
        print("Question Paper Generator Is Loading.....")
        titleLabel = Label(my_fill, text='Fill in the Blanks Generator ', font=('italic', 25, 'bold '), bg='black',fg='DarkTurquoise')
        titleLabel.place(x=70, y=8,width=1100, height=80)




        def result():
            text =upload_file()
            
            def tokenize_sentences(text):
                    sentences = sent_tokenize(text)
                    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
                    return sentences
            sentences = tokenize_sentences(text)


            def get_noun_adj_verb(text):
                out=[]
                try:
                    extractor = pke.unsupervised.MultipartiteRank()
                    extractor.load_document(input=text,language='en')
                    pos = {'VERB', 'ADJ', 'NOUN'}
                    stoplist = list(string.punctuation)
                    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
                    stoplist += stopwords.words('english')
                    extractor.candidate_selection(pos=pos)
                    
                    extractor.candidate_weighting(alpha=1.1,
                                                    threshold=0.75,
                                                    method='average')
                    keyphrases = extractor.get_n_best(n=30)
                    

                    for val in keyphrases:
                        out.append(val[0])
                except:
                    out = []
                    traceback.print_exc()

                return out

            noun_verbs_adj = get_noun_adj_verb(text)

            from pprint import pprint
            def get_sentences_for_keyword(keywords, sentences):
                keyword_processor = KeywordProcessor()
                keyword_sentences = {}
                for word in keywords:
                    keyword_sentences[word] = []
                    keyword_processor.add_keyword(word)
                for sentence in sentences:
                    keywords_found = keyword_processor.extract_keywords(sentence)
                    for key in keywords_found:
                        keyword_sentences[key].append(sentence)

                for key in keyword_sentences.keys():
                    values = keyword_sentences[key]
                    values = sorted(values, key=len, reverse=True)
                    keyword_sentences[key] = values
                return keyword_sentences

            keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
            


            def get_fill_in_the_blanks(sentence_mapping):
                out={"title":"Fill in the blanks for these sentences with matching words at the top"}
                blank_sentences = []
                processed = []                                                  
                keys=[]
                for key in sentence_mapping:
                    if len(sentence_mapping[key])>0:
                        sent = sentence_mapping[key][0]
                        insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                        no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
                        line = insensitive_sent.sub(' _________ ', sent)
                        if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
                            blank_sentences.append(line)
                            processed.append(sentence_mapping[key][0])
                            keys.append(key)
                out["sentences"]=blank_sentences[:21]
                a=out["sentences"]
            
                return a

           

            
            print("Question paper is Loading...!") 

            pdf = fpdf.FPDF(format='letter')
            pdf.add_page()
            pdf.add_font('Arial', '', 'c:/windows/fonts/arial.ttf', uni=True)
            pdf.set_font("Arial", size=30)
            pdf.write(30,"Question Paper")
            pdf.ln()
            pdf.set_font("Arial", size=12)
            quesful=[]
            fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
            
            b=fill_in_the_blanks
            c=[]
            for i in range(len(b)):
                a=str(i+1)+str(") ")+b[i]
                c.append(a)
            pdf = fpdf.FPDF(format='letter')
            pdf.add_page()
            pdf.add_font('Arial', '', 'c:/windows/fonts/arial.ttf', uni=True)
            pdf.set_font("Arial", size=30)
            pdf.write(30,"Question Paper")
            pdf.ln()
            pdf.set_font("Arial", size=12)
            for i in c:
                pdf.write(10,str(i))
                pdf.ln(15)
            pdf.output("questionpaper_fillup.pdf")

            



        
            return pdf



        def qsnpaper():
            finalprint =result()
            l3 = tk.Label(my_fill,text='\n Generated Question Paper are \n successfully saved as \n \n',font=('italic', 20, 'bold '),
                        fg='white',bg="black")  
            l3.place(x=350, y=250,width=420)
            l5 = tk.Label(my_fill,text='questionpaper_fillup.pdf',font=('italic', 20, 'bold '),
                        fg='DarkTurquoise',bg="black")  
            l5.place(x=392, y=350)
        
            #----------SEE PDF Button----------------------------------------------------------------------------------------

            b2 = tk.Button(my_fill, text='SEE PDF', 
                width=20,command = lambda:see_pdf(), activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='DarkTurquoise')
            b2.place(x=470,y=600, width=155, height=40)

            #------------------------------------------------------------------------------------------------------------

            #----------CLOSE Button----------------------------------------------------------------------------------------

            b3 = tk.Button(my_fill, text='CLOSE', 
                width=20,command = lambda:close(my_fill), activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='yellow')
            b3.place(x=790,y=600, width=140, height=40)

            #------------------------------------------------------------------------------------------------------------
      
            return 0    


        def upload_file():
            l10 = tk.Label(my_fill,text='\n Wait ! \n Questions are Generating \n',width=45,font=('italic',20,'bold'),bg='DarkTurquoise',
                        fg='black',)  
            l10.place(x=350, y=250,width=420)
            
            f_types = [('Text Files', '*.txt'),('Doc Files','*.docx')]   
            filename = tk.filedialog.askopenfilename(multiple=False,filetypes=f_types)

            if filename:
                try:
                    with open(filename, "r+", encoding='utf-8') as file:
                        text_content = file.read()
                except UnicodeDecodeError as error:
                    with open(filename, "r+", encoding='ISO-8859-1') as file:
                        text_content = file.read()
                return text_content
            else:
                messagebox.showwarning("Alert", "No file selected!")
                return None



        def see_pdf():

            def save_pdf_with_password(input_path, output_path, password):
                pdf = pikepdf.open(input_path, allow_overwriting_input=True)
                pdf.save(output_path, encryption=pikepdf.Encryption(user=password))
                pdf_location=r"C:/Users/prajwalitha/Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaper2.pdf"
                webbrowser.open(pdf_location)

            
            input_pdf_path = "C:/Users/prajwalitha/Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaper_fillup.pdf"
            output_pdf_path = "C:/Users/prajwalitha/Desktop/Question paper generator project/Question paper generator project/Tkinter---Question paper generator/questionpaper2.pdf"
            password = "exams"
            save_pdf_with_password(input_pdf_path, output_pdf_path, password)

    def close(window):
        window.destroy()

    l1 = tk.Label(my_w,text='Question Paper Generator',width=45,font=('italic',30,'bold'),bg='black',
                       fg='white',)  
    l1.place(x=290, y=10,width=500)


    btn5 = tk.Button(my_w, text="Generate Question", command=qsngenerator, activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='yellow')   
    btn5.place(x=250, y=300,width=220)


    btn2=tk.Button(my_w, text="Generate Fill up", command=fillupgen, activebackground='skyblue',font=('italic',17,'bold') ,bg='black',fg='yellow')   
    btn2.place(x=550, y=300,width=200)



    l2 = tk.Label(my_w,text='Note: After clicking the buttons \n the Pages will open in a next window',width=45,font=('italic',12,'bold'),bg='black',
                       fg='grey',)  
    l2.place(x=5, y=655,width=300,height=50)



    my_w.mainloop()








root = Tk()
root.geometry('1350x710+0+10')  
root.title('Login Page')
my_font1=('times', 18, 'bold')

bg = PhotoImage(file="backgroundpic.png")
bgLabel = Label(root,image=bg)
bgLabel.place(x=0, y=0)


con = sqlite3.connect('datalogin.db')
cur = con.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS userregister(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, mail TEXT, password TEXT,status TEXT)')
con.commit()
con.close

def user_login():
    login_bg = 'lavender'
    global User_login_Frame
    User_login_Frame = Frame( padx=70, pady=50 , bg=login_bg )
    Label(User_login_Frame,text='User Login' , font= ('Arial',22,'bold'), bg='white', fg='red' ).pack(pady=10)

    
    Label(User_login_Frame,text='Email',textvariable='email' , font=('times new roman', 18, 'bold'),underline= 1,bg ='black',fg='white').pack(pady=10)
    entry01 = tk.Entry(User_login_Frame, font=('times new roman', 18), bg='lightgray')
    entry01.pack()
    Label(User_login_Frame,text='Password' , font=('times new roman', 18, 'bold'),underline= 1,bg ='black',fg='white').pack(pady=10)
    entry02 = tk.Entry(User_login_Frame,show='*',font=('times new roman', 18), bg='lightgray')
    entry02.pack()
    

    def login_Close():
        if entry01.get() != '' and entry02.get() != '' :
            cur.execute("select * from userregister where mail=? and Password=?",(entry01.get(),entry02.get()))
            row = cur.fetchone()
         
            if row != None:
                question()
               
                
            else:
                messagebox.showerror('Failed','Login Failed.')
        else:
            messagebox.showwarning("Alert","Enter Email & Password Correctly !!")

    def open_Register():
        User_login_Frame.destroy()
        user_Register()
    

    tk.Button(User_login_Frame,text='Login ✔' ,bg='green',command=login_Close).pack(pady=10)
    Button(User_login_Frame,text='Not Registered ?' , bd= 0 , bg=login_bg , relief='flat' , overrelief='flat' , command=open_Register ).pack(pady=10)
    User_login_Frame.pack(pady=50)
   
def user_Register():
    register_bg = 'lightyellow'
    register_Frame = Frame( padx=70, pady=50, bg = register_bg )
    Label(register_Frame, text='Register For User' , font= ('Arial',22,'bold') , bg ='white',fg='red' ).pack(pady=10)

    Label(register_Frame, text='Name', font=('times new roman', 18, 'bold') ,underline= 1,bg ='white',fg='blue' ).pack(pady=10)
    reg_entry01 = tk.Entry(register_Frame, font=('times new roman', 18), bg='white')
    reg_entry01.pack()
    Label(register_Frame, text='Email', font=('times new roman', 18, 'bold'),underline=1,bg ='white',fg='blue'  ).pack(pady=10)
    reg_entry02 = tk.Entry(register_Frame, font=('times new roman', 18), bg='white')
    reg_entry02.pack()
    tk.Label(register_Frame, text='Password',font=('times new roman', 18, 'bold'),underline=1,bg ='white',fg='blue'  ).pack(pady=10)
    reg_entry03 = tk.Entry(register_Frame , show='*', font=('times new roman', 18), bg='white' )
    reg_entry03.pack()
    tk.Label(register_Frame, text='Re-Enter Password', font=('times new roman', 18, 'bold'),underline=1,bg ='white',fg='blue' ).pack(pady=10)
    reg_entry04 = tk.Entry(register_Frame , show='*', font=('times new roman', 18), bg='white' )
    reg_entry04.pack()
    def goback():
        register_Frame.destroy()
        user_login()

        
    def register():
        if reg_entry01.get() != '' and reg_entry02.get() != '' and reg_entry03.get() != '':
            if reg_entry03.get() == reg_entry04.get():
                cur.execute('insert into userregister(name,mail,password,status) values(?,?,?,?)',(reg_entry01.get(),reg_entry02.get(), reg_entry03.get(), reg_entry04.get()))
                con.commit()
                
                register_Frame.destroy()
                messagebox.showinfo('Success','Registered Successfully.')
                user_login()
            else:
                messagebox.showerror("Error","Password and Re-Enter Password doesn\'t Match.")
        else:
            messagebox.showerror('Error','Enter Name,Email and Password Correctly.')

    tk.Button( register_Frame , text='Register' ,bg='green',command=register ).pack(pady=10)
    register_Frame.pack(pady=20)
    tk.Button(register_Frame, text='GoBack' ,bg='green', command=goback).pack(pady=20)
    register_Frame.pack(pady=50)


user_login()
   


                                                                                                                                
