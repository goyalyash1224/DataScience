# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:24:32 2022

@author: yash
"""


#import the libraries
import pickle
import time
from time import sleep
import logging
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import os


#import global variables
start_time = time.time()
person = []
date = []
stars = []
review = []
sentiment = []


#define export_data function to export data in list variables
def export_data():
    # if 'scrappedReviews.csv' in os.listdir(os.getcwd()):
    #     data = pd.read_csv('scrappedReviews.csv')
    #     '''exporting the data'''
        
    
    #     for i in range(0,len(data["start"])):
    #         if data["Person"][i] not in person:
    #             person.append(data["Person"][i])
    #             date.append(data["Date"][i])
    #             stars.append(data["Stars"][i])
    #             review.append(data["Review"][i])
    #             sentiment.append(data["Sentiment"][i])
                
    #         dataframe1 = pd.DataFrame()
    #         dataframe1['Person'] = person
    #         dataframe1['Date'] = date
    #         dataframe1['Stars'] = stars
    #         dataframe1['Review'] = review
    #         dataframe1['Sentiment'] = sentiment
        
    #         result = pd.concat([data,dataframe1])
    #         result.to_csv('scrappedReviews.csv', index=False)
            
    # else:
    ''' exporting the data '''
    dataframe1 = pd.DataFrame()
    dataframe1['Stars'] = stars
    dataframe1['Review'] = review
    dataframe1['Sentiment'] = sentiment

    dataframe1.to_csv('scrappedReviews.csv',mode='a',header=False, index=False)
    stars.clear()
    review.clear()
    sentiment.clear()
        
#defining the function to check the sentiment of any review        
def check_review(reviewText):
    
    ''' check the review is positive or negative '''
    
    file = open('pickle_model.pkl','rb')
    pickle_model = pickle.load(file)
    file = open('features.pkl','rb')
    vocab = pickle.load(file)
    
    #reviewText has to vectorised, and to convert it in vector we first load vectorizer then using strored features we transform it
    #Now using model we can predict sentiment of the reviewText
    
    transformer = TfidfTransformer()
    loaded_vec  = CountVectorizer(decode_error='replace',vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    
    #predict the sentiment using pickle_model
    result = pickle_model.predict(vectorised_review)
    
    return result[0]

def run_scrapper(page):
    global person, review
    print('starting chrome')
    
    #install chromeDriver and load it in a variable
    s = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=s)
    # browser = webdriver.Chrome(ChromeDriverManager().install())
    
    URL ='https://www.etsy.com/in-en/c/jewelry-and-accessories?ref=pagination&page={}'
    
    
    '''
    # # Test Code for a page and content on that page  
    # URL = URL.format(11)
    # browser.get(URL)
    # # sleep(1)
    # # print("Scrapping the page :", page)
    # #xpath to product table
    # # PATH_1 = '/html/body/div[5]/div/div[1]/div[3]/div[2]/div[2]/div[9]/div/div/div/ul'
    # # PATH_1 = '//*[@id="content"]/div[1]/div[1]/div[1]/div[3]/div[2]/div[2]/div[5]/div[1]/div[1]/div[1]/div[1]/ul'
    # PATH_1 = '//*[@id="content"]/div/div/div/div[3]/div[2]/div[2]/div[5]/div/div/div/ul'
    # #getting total items
    # items = browser.find_element(By.XPATH,PATH_1)
    # print(items)
    # items = items.find_elements(By.TAG_NAME,'li')
    # print(len(items))
    # # print(items)
    # #available itmes in the page
    # # end_item = len(items)
    # # print(end_item)
    # link = items[0].find_element(By.TAG_NAME,"a")
    # print(link.click())
    # print("success")
    
    # #switch the focus of driver to ne|w tab
    # windows = browser.window_handles
    # browser.switch_to.window(windows[1])
    # print("success")
    # count = browser.find_element(By.XPATH,'//*[@id="reviews"]/div[2]/div[2]')
    # count = count.find_elements(By.CLASS_NAME,'wt-grid__item-xs-12')
    # print(count)
    # print("Double Success")
    # dat1 = browser.get_element(By.XPATH,
    #             '//*[@id="reviews"]/div[2]/div[2]/div[{}]/div/p'.format(2)).text
    # print(dat1)
    # print("tiple success")
    # browser.close()
    '''
      
    
    try:   
        URL = URL.format(page)
        browser.get(URL)
        
        sleep(1)
        print("Scrapping the page :", page)
        #xpath to product table
        # PATH_1 = '/html/body/div[5]/div/div[1]/div/div[4]/div[2]/div[2]/div[3]/div/div/ul'
        # PATH_1 = '//*[@id="content"]/div[1]/div[1]/div[1]/div[3]/div[2]/div[2]/div[5]/div[1]/div[1]/div[1]/ul'
        PATH_1 = '//*[@id="content"]/div/div/div/div[3]/div[2]/div[2]/div[5]/div/div/div/ul'
        #getting total items
        items = browser.find_element(By.XPATH,PATH_1)
        items = items.find_elements(By.TAG_NAME,'li')
        # print(items)
       
        #available itmes in the page
        end_item = len(items)
        # print("Total Items on this page: ", end_item)
        #count for every product on the page
        for product in range(0,end_item):
            if(product%10==0):
                export_data()
            print(' Scrapping reviews for product :', product+1 )
            print(items[product])
            
            try:
                items[product].find_element(By.TAG_NAME,'a').click()
            except:
                print("Product link not found")
            
            # print("success")
            #switch the focus of driver to ne|w tab
            windows = browser.window_handles
            browser.switch_to.window(windows[1])
            
            # sleep(1)
            try:
                # print("TRY  1  \n")
                
                PATH_2 = '//*[@id="same-listing-reviews-panel"]/div'
                count = browser.find_element(By.XPATH,PATH_2)
                
                #Number of review on that page
                count = count.find_elements(By.CLASS_NAME,'wt-grid__item-xs-12')
                for r1 in range(1, len(count)+1):
                    data1 = browser.find_element(By.XPATH,
                        '//*[@id=same-listing-reviews-panel]/div/div[{}]/div[2]/p[1]'.format(r1)).text
                    
                    if data1[:data1.find(',')-6] not in person:
                        try :
                            person.append(data1[:data1.find(',')-6])
                            date.append(data1[data1.find(',')-6:])
                        except Exception:
                            person.append("Not Found")
                            date.append("Not Fount")
                        try:
                            stars.append(browser.find_element(By.XPATH,
                                '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[2]/div/div/div[1]/span/span[1]'.format(
                                    r1)).text[0])
                        except Exception:
                            stars.append("No stars")
                        
                        try:
                            review.append(browser.find_element(By.XPATH,
                                '//*[@id="review-preview-toggle-{}"]'.format(r1-1)).text)
                            sentiment.append(check_review(browser.find_element(By.XPATH,
                                '//*[@id="review-preview-toggle-{}"]'.format(r1-1)).text))
                        except Exception:
                            review.append('No Review')
                            sentiment.append(check_review('No Review'))
                    browser.close()
            
            except Exception:
                # print("TRY  2  \n")
               
                count = browser.find_element(By.XPATH,'//*[@id="reviews"]/div[2]/div[2]')
                count = count.find_elements(By.CLASS_NAME,'wt-grid__item-xs-12')
                print("Total Review for this product:" , len(count))
               
                for r2 in range(1,len(count)+1):
                    # print("enter in try 2")
                    # dat1 = browser.find_element(By.XPATH,
                    #             '//*[@id="reviews"]/div[2]/div[2]/div[{}]/div[1]/p'.format(r2))
                    # dat1 =  dat1.text
                    
                    # if dat1[:dat1.find(',')-6] not in person:
                    # print("enter in loop of try 2")
                    # try:
                        
                    #     person.append(dat1[:dat1.find(',')-6])
                    #     date.append(dat1[dat1.find(',')-6:])
                    # except Exception:
                    #     person.append("Not Found")
                    #     date.append("Not Found")
                    try:
                        stars.append(browser.find_element(By.XPATH,
                            '//*[@id="reviews"]/div[2]/div[2]/div[{}]/div[2]/div[1]/div[1]/div[1]/span/span[1]'.format(
                                r2)).text[0])
                    except Exception:
                        stars.append("No Stars")
                        print("..........no stars.............")
                    try:
                        review.append(browser.find_element(By.XPATH,
                            '//*[@id="review-preview-toggle-{}"]'.format(
                                r2-1)).text)
                        
                        sentiment.append(check_review(
                            browser.find_element(By.XPATH,
                            '//*[@id="review-preview-toggle-{}"]'.format(
                                r2-1)).text))
                        print("FOUND REVIEW")
                    except Exception:
                        review.append("No Review")
                        sentiment.append(check_review(
                            "No Review"))
                        print("............NOT FOUND REVIEW.............")
                        
                  
            except Exception:
                    try:
                        print("TRY  3  \n")
                        count = browser.find_element(By.XPATH,'//*[@id="reviews"]/div[2]/div[2]')
                        count = count.find_elements(By.CLASS_NAME,'wt-grid__item-xs-12')
                        
                        for r3 in range(1,len(count)+1):
                            dat1 = browser.find_element(By.XPATH,
                                        '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[1]/p'.format(r3)).text
                            if dat1[:dat1.find(',')-6] not in person:
                                try:
                                    person.append(dat1[:dat1.find(',')-6])
                                    date.append(dat1[dat1.find(',')-6:])
                                except Exception:
                                    person.append("Not Found")
                                    date.append("Not Found")
                                try:
                                    stars.append(browser.find_element(By.XPATH,
                                        '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[2]/div[1]/div[1]/div[1]/span/span[1]'.format(r3)).text[0])
                                except Exception:
                                    stars.append("No Stars")
                                try:
                                    review.append(browser.find_element(By.XPATH,
                                        '//*[@id="review-preview-toggle-{}"]'.format(r3-1)).text)
                                    sentiment.append(check_review(browser.find_element(By.XPATH,
                                        '//*[@id="review-preview-toggle-{}"]'.format(r3-1)).text))
                                except Exception:
                                    review.append("No Review")
                                    sentiment.append(check_review("No Review"))
                        

                    except Exception:
                        print("Error")
                        continue
            browser.close()
                                
            
            #switch focus to main tab
            browser.switch_to.window(windows[0])
        
        
    except Exception as e_1:
        print(e_1)
        print("Program stopped")
    export_data()
    browser.close()
    
    
    
#defining the main function

def main():
    logging.basicConfig(filename='solution_etsy.log', level=logging.INFO)
    logging.info('Started')
    
    if 'page.txt' in os.listdir(os.getcwd()):
        with open('page.txt','r') as file1:
            page = int(file1.read())
        for i in range(page,251):
            run_scrapper(i)
    else:
        for i in range(1,251):
            with open('page.txt','w') as file:
                file.write(str(i))
            run_scrapper(i)
        
    export_data()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    logging.info('Finished')
        
# Calling the main function 
if __name__ == '__main__':
    main()

        