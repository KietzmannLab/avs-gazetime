# This script will be the basis for a GUI that presents the user with anny given AVS scene and asks them to check whether a given fixation target is mirrored in the scene description.

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import argparse
import tkinter as tk
from PIL import ImageQt
import io
import spacy # this migh require running "python -m spacy download de_core_news_sm" in the terminal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QPushButton, QGridLayout, QScrollArea, QFrame, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QT_VERSION_STR
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
from tkinter import Tk, Label, Button, Entry, Listbox, END
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QLineEdit, QPushButton

print(QT_VERSION_STR)
# enable qt debug mode. set to 1 to enable
os.environ['QT_DEBUG_PLUGINS'] = '0'

# import custom AVS functions 


class Fixation2CaptionMatcher():
    def __init__(self, data_dir=None, screen_usage = 0.925,
                 stim_screen_size_xy=(1024, 768), input_image_size_xy=(947,710), used_screen_area = 0.925):
        self.et_dir = os.path.join(data_dir, 'et')
        self.scene_dir = os.path.join(data_dir, 'scenes')
        self.data_dir = data_dir
        self.tk_geometry = "800x800"
        self.start = False
        self.screen_usage = screen_usage
        self.stim_screen_size_xy = stim_screen_size_xy
        self.input_image_size_xy = input_image_size_xy
        self.screen_y_pix = self.stim_screen_size_xy[1]
        self.screen_x_pix = self.stim_screen_size_xy[0]
        self.used_screen_area = used_screen_area
      
        # import spacy
        print("Loading spacy model. This might take a while.")
        self.nlp = spacy.load("de_core_news_sm")
        print("Spacy model loaded.")

    def get_et_data(self, user_id):
        """It opens the eye tracking data frame. it looks whether we already have a caopy of that df from that user. If not it will create one."""
        # check whether we already have a copy of the df
        et_fname_orig = os.path.join(self.et_dir, 'fix2cap_events.csv')
        et_fname_copy = os.path.join(self.et_dir, 'fix2cap_events_' + user_id + '.csv')
        if os.path.isfile(et_fname_copy):
            # if we have a copy, load it
            et_df = pd.read_csv(et_fname_copy, low_memory=False)
            # remove al columns that have Unnamed in the name
            et_df = et_df.loc[:, ~et_df.columns.str.contains('^Unnamed')]
        else:
            # if we don't have a copy, load the original
            et_df = pd.read_csv(et_fname_orig, low_memory=False)
            # and save it under the user id
            # add the fixation2caption columns [in_caption, in_context]
            # add an empty string for the context word
            et_df['context_word'] = pd.Series(np.zeros(len(et_df)), index=et_df.index, dtype=str)
            # add an empty column for the fix target word
            et_df['fix_word'] = pd.Series(np.zeros(len(et_df)), index=et_df.index, dtype=str)
            et_df['in_caption'] = False
            et_df['none_from_other_subject'] = pd.Series(np.zeros(len(et_df)), index=et_df.index, dtype=str)
            et_df['none_typed'] = pd.Series(np.zeros(len(et_df)), index=et_df.index, dtype=str)
            et_df['fix2cap'] = False
            et_df['fix2cap_done'] = False
            # # check wich rows have a "time_in_trial" value of that is not empty those get a fix2cap value of True
            et_df.loc[et_df['time_in_trial'].notnull(), 'fix2cap'] = True  # this will tell us that those events have to be checked
            et_df.to_csv(et_fname_copy)
        self.et_df = et_df


    def get_scene_image(self, sceneID):
        im = Image.open(self.scene_dir + os.sep + str(int(sceneID)).zfill(12) + '_MEG_size.jpg')

        # get the measures of the scene
        im_width = im.width
        im_height = im.height
        # get and resize the scene to the size it had during the presentation

        im_scaler = (self.screen_y_pix * self.screen_usage) /im_height
        # only scale if if imscaler rounds to 1 after 2 decimal places
        
        if np.round(im_scaler, 2) != 1:
            print("Scaling scene by {}".format(im_scaler))
        
            im_width_rescaled = int(im_width * im_scaler)
            im_height_rescaled = int(im_height * im_scaler)
            
        
            im_rescaled = im.resize((im_width_rescaled,im_height_rescaled) )
        else:
            print("No scaling necessary")
            im_rescaled = im
            im_width_rescaled = im_width
            im_height_rescaled = im_height
        return im_rescaled, im_width_rescaled, im_height_rescaled
    
    def mark_fixation_target(self, x, y , im_rescaled, im_width_rescaled, im_height_rescaled, color = "magenta", s=50, words = None, word_color = "white"):
        # x and y also can be a list of x and y coordinates
        # words can be a list of words that will be written at the x and y coordinates
        # word_color is the color of the words, it can be a list of colors

        # add a circle around the fixation target

        

        draw = ImageDraw.Draw(im_rescaled)

        # center the coordinates
        x =  x - (self.screen_x_pix/2)
        y =  y - (self.screen_y_pix/2)

        # draw a circle
        fig, ax_fix = plt.subplots()
         # plot the image to the center of the coordinate system
        ax_fix.imshow(im_rescaled,  extent=(-im_width_rescaled/2, im_width_rescaled/2, -im_height_rescaled/2, im_height_rescaled/2))
        ax_fix.scatter(x, y, s=s, c=color, marker="+")
        # add a round circle around the fixation target
        
        if words is not None:
            # assert that the number of words is the same as the number of x and y coordinates
            assert len(words) == len(x)
            # if word_color is not a list, make it a list
            if not isinstance(word_color, list):
                word_color = [word_color] * len(x)


            for i, (x_i, y_i) in enumerate(zip(x, y)):
                ax_fix.text(x_i, y_i, words[i], fontsize=12, color=word_color[i], ha="right", va="center")
        else:
            circle = plt.Circle((x, y), 50, color=color, fill=False)
            ax_fix.add_artist(circle)
        # axis off
        ax_fix.axis('off')
        # tight layout
        fig.tight_layout()

        # write theimage to a temp dir 
        temdir = os.path.join(self.data_dir, 'temp')
        if not os.path.isdir(temdir):
            os.mkdir(temdir)
        fname = os.path.join(temdir, 'fixation_target.png')
        fig.savefig(fname)
        # open the image again

        im_rescaled = fname
        plt.close(fig)

        # return the image
        return im_rescaled


    
    def run_welcome_screen(self):
        """It welcomes the user and asks them for their user id."""
        # create a window
        self.welcome_window = tk.Tk()
        # set window title
        self.welcome_window.title("Welcome to the AVS fixation2caption matcher!")
        # set window size
        self.welcome_window.geometry(self.tk_geometry)
        # add a label to the window
        self.welcome_label = tk.Label(self.welcome_window, text="Welcome to the AVS fixation2caption matcher!")
        # add a label to the window
        self.user_label = tk.Label(self.welcome_window, text="Please enter your user ID:",)
        # add an entry field to the window (pre_fill the textbox with user "test")
        self.user_entry = tk.Entry(self.welcome_window, width=10)
        self.user_entry.insert(0, "test12")
        # pack the label to the window
        self.welcome_label.pack()
        # pack the label to the window
        self.user_label.pack()
        # pack the entry field to the window
        self.user_entry.pack()


        def get_id(self):
            user_id = self.user_entry.get()
            # once id is not empty, close the window
            if user_id != "":
                self.welcome_window.destroy()
            else:
                # if the user id is empty, display a warning
                self.user_label.config(text="Please enter your user ID:", fg="red")

            print(user_id)
            self.user_id = user_id

        
        # add a button to the window 
        # also allow the user to press enter to submit the user id
        self.user_button = tk.Button(self.welcome_window, text="Submit", command=get_id)
        self.welcome_window.bind('<Return>', lambda event = None: get_id(self))
        # pack the button to the window
        self.user_button.pack()


        # run the window
        self.welcome_window.mainloop()
        return self.user_id

  

    
    def display_progress(self, user_id):
        """It displays the progress of the user. It will show the user how many trials they have already completed per subject and session."""
        # we will lock at the et data frame and check how many trials are already done
        df_progress = self.et_df.groupby(['subject'])['fix2cap_done'].sum().reset_index()
        # print the column names
        print(df_progress.columns)

        # compute the percentage of trials that are done
        df_progress['length'] = self.et_df.groupby(['subject'])['fix2cap_done'].count().reset_index()['fix2cap_done']
        df_progress['percentage'] = np.round(df_progress['fix2cap_done'] / df_progress['length'],2  ) * 100
        # remove the length column
        df_progress = df_progress.drop(columns=['length'])
        # remove the fix2cap_done column
        df_progress = df_progress.drop(columns=['fix2cap_done'])
        # only show the last two columns
        df_progress = df_progress[['subject', 'percentage']]

        # display the progress in the gui
        # create a window
        self.progress_window = tk.Tk()
        # set window title
        self.progress_window.title("Your progress")
        # set window size
        self.progress_window.geometry(self.tk_geometry)
        # add a label to the window
        self.progress_label = tk.Label(self.progress_window, text="Your progress:")
        # add a label to the window
        self.progress_text = tk.Label(self.progress_window, text=df_progress)
        # pack the label to the window
        self.progress_label.pack()
        # pack the label to the window
        self.progress_text.pack()
        # # add a start button to the window (this will be a continue button)

        def continue_fix2cap_matcher():
            self.progress_window.destroy()
            return

        self.progress_button = tk.Button(self.progress_window, text="Continue", command=continue_fix2cap_matcher)
        # also allow the user to press enter to continue
        self.progress_window.bind('<Return>', lambda event = None: continue_fix2cap_matcher())
        # pack the button to the window
        self.progress_button.pack()
        # run the window
        self.progress_window.mainloop()
        return
    
    
    def get_current_event(self):
        """It gets the current event that has to be checked."""
        # get the earliest unchecked event (based on min subject , min session number, min trial number, min fix_position)
        unchecked_events = self.et_df.loc[self.et_df['fix2cap'] == True]
        # done should be false
        unchecked_events = unchecked_events.loc[unchecked_events['fix2cap_done'] == False]
        if len(unchecked_events) == 0:
            # if there are no unchecked events left, quit the script
            print("No unchecked events left. Hefty congrats! You are done!")
            self.run_matcher = False
            self.app.quit()
            return
        min_subject = unchecked_events['subject'].min()
        min_session = unchecked_events.loc[unchecked_events['subject'] == min_subject]['session'].min()
        min_trial = unchecked_events.loc[(unchecked_events['subject'] == min_subject) & (unchecked_events['session'] == min_session)]['trial'].min()
        min_fix_position = unchecked_events.loc[(unchecked_events['subject'] == min_subject) & (unchecked_events['session'] == min_session) & (unchecked_events['trial'] == min_trial)]['fix_sequence'].min()
        self.unchecked_event = unchecked_events.loc[(unchecked_events['subject'] == min_subject) & (unchecked_events['session'] == min_session) & (unchecked_events['trial'] == min_trial) & (unchecked_events['fix_sequence'] == min_fix_position)]
        # check if this is the frist evenbt of this scene
        min_fixpos_this_scene = min_trial = unchecked_events.loc[(unchecked_events['subject'] == min_subject) & (unchecked_events['session'] == min_session) & (unchecked_events['trial'] == min_trial)]['fix_sequence'].min()
        if self.unchecked_event['fix_sequence'].values[0] == min_fixpos_this_scene:
            # if this is the first event of this scene, get the scene image
            first_event_this_scene = True
        return first_event_this_scene
    
    def get_scene_with_fixation_target(self,):
        """It gets the scene image with the fixation target marked."""
        # get the scene image
        scene_image, scene_image_width, scene_image_height = self.get_scene_image(self.unchecked_event['sceneID'].values[0])
         # check if the current event is already has a context word that is not empty or "None"
        print("context word: {}".format(self.unchecked_event['context_word'].values[0]))
        # get all the events of the current subject, session, trial
        events_scene = self.et_df.loc[(self.et_df['subject'] == self.unchecked_event['subject'].values[0]) & (self.et_df['session'] == self.unchecked_event['session'].values[0]) & (self.et_df['trial'] == self.unchecked_event['trial'].values[0])]
        # do we have a context word for this scene that is not 0.0?
        display_context_word_window = True
        for context_word in events_scene['context_word']:
            if context_word != "0.0":
                display_context_word_window = False
                break
        if display_context_word_window:
            # wrtie the scene image to a temp dir
            temdir = os.path.join(self.data_dir, 'temp')
            if not os.path.isdir(temdir):
                os.mkdir(temdir)
            fname = os.path.join(temdir, 'scene_image.png')
            # rescale to 50% of the original size
            scene_image = scene_image.resize((int(scene_image_width * 0.5), int(scene_image_height * 0.5)))
            scene_image.save(fname)
            # if this is the first event of this scene, display a message that the user has to check whether the fixation target is mirrored in the scene description
            self.get_context_word_window(fname)
        # draw the fixation target on the scene image
        scene_image_with_fix = self.mark_fixation_target(self.unchecked_event['mean_gx'].values[0], self.unchecked_event['mean_gy'].values[0], scene_image, scene_image_width, scene_image_height)
       

        return scene_image_with_fix
    
    def get_caption_nouns(self, caption):
        """It gets the nouns from the caption."""
        # get the nouns from the caption
        doc = self.nlp(caption)
        nouns = []
        for token in doc:
            if token.pos_ == "NOUN":
                nouns.append(token.text)
        return nouns
    
    def store_context_word(self, selected_word):
        # get the answer
        print("store context word")
        # close the window
        # store the answer in the et_df
        self.unchecked_event_index = self.unchecked_event.index[0]
        if selected_word != "None" and selected_word != "0.0":
            print("selected context word: {}".format(selected_word))
            # set all events of that subject, session, trial ["context_word"] to the selected word
            current_subject = self.et_df.loc[self.unchecked_event_index, 'subject']
            current_session = self.et_df.loc[self.unchecked_event_index, 'session']
            current_trial = self.et_df.loc[self.unchecked_event_index, 'trial']
            self.et_df.loc[(self.et_df['subject'] == current_subject) & (self.et_df['session'] == current_session) & (self.et_df['trial'] == current_trial), 'context_word'] = selected_word
            # also set the context word of the current event
            self.unchecked_event['context_word'] = selected_word
        else:
            print("selected context word: None")
            self.et_df.loc[self.unchecked_event_index, 'context_word'] = "no context word"
        # save the et_df
        print(selected_word)
        #self.et_df.to_csv(os.path.join(self.et_dir, 'fix2cap_events_' + self.user_id + '.csv'))

        self.context_word_window.close()
        # quit the application
        return 
    
    def get_context_word_window(self, sene_image_raw):
        """1) Show the caption sentence 2)show the image witho the fixation target 3) ask the user to choose the caption noun that describes the context/location of the fixation target."""

        # create the self.application
        print("create application")
        # wait for the creation of the self.application
        
        self.unchecked_event_index = self.unchecked_event.index[0]

        current_subject = self.unchecked_event['subject'].values[0]
        current_session = self.unchecked_event['session'].values[0]
        current_trial = self.unchecked_event['trial'].values[0]
        current_fix_sequence_position = self.unchecked_event['fix_sequence'].values[0]
        current_caption = self.unchecked_event['transcribed_caption'].values[0]
        # set window title
        self.context_word_window = QWidget()
        # set fontsize to 16 with style sheet
        self.context_word_window.setStyleSheet("font-size: 16pt")
        print("create window")

        self.context_word_window.setWindowTitle("Fix2cap matcher: Subject {}, Session {}, Trial {}, Fix {}".format(current_subject, current_session, current_trial, current_fix_sequence_position))

        # set window size use 1200 x 900
        self.context_word_window.setGeometry(50, 50, 800, 600)

        layout = QVBoxLayout()
        

        # add a label to the window
        context_word_label = QLabel("NEW CAPTION: Identify the noun that describes the context/location of the fixation target.")
        # make the label blue
        context_word_label.setStyleSheet("QLabel {color: blue}")

        layout.addWidget(context_word_label)

        # display the caption in bold helvetica
        caption_label = QLabel(current_caption)
        print(current_caption)
        caption_label.setFont(QFont("Helvetica", 16, QFont.Bold))
        layout.addWidget(caption_label)

        # Display the scene image
        scene_image_label = QLabel()
        # read the scene image as a QImage
        # show the scene image
        plt.imshow(Image.open(sene_image_raw))

        scene_qimage = QImage(sene_image_raw)
        # plt show the scene image
        # convert the QImage to a QPixmap
        #scene_pixmap = QPixmap.fromImage(scene_qimage)
        # set the scene image label to the scene pixmap
        scene_image_label.setPixmap(QPixmap.fromImage(scene_qimage))
        # add the labels to the window
        layout.addWidget(scene_image_label)

        word_buttons_frame = QGroupBox()
        word_buttons_layout = QVBoxLayout()
        word_buttons_frame.setLayout(word_buttons_layout)

        # Add buttons for each noun
        row_layout = QHBoxLayout()  # Create a new QHBoxLayout for each row of buttons
        nouns = self.get_caption_nouns(current_caption)
       
        for i, noun in enumerate(nouns):
            print(noun)
            button_text = noun + " (" + str(i + 1) + ")"
            noun_button = QPushButton(button_text)
            noun_button.clicked.connect(lambda checked, word=noun: self.store_context_word(word))
            row_layout.addWidget(noun_button)

            # Add a new row layout after every 4 buttons
            if (i + 1) % 4 == 0:
                word_buttons_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()

        # Add the remaining buttons in the last row
        if row_layout.count() > 0:
            word_buttons_layout.addLayout(row_layout)

        # Add a "None" button
        none_button = QPushButton("None (0)")
        none_button.clicked.connect(lambda checked, word="None": self.store_context_word(word))
        word_buttons_layout.addWidget(none_button)
        # add the buttons to the window
        layout.addWidget(word_buttons_frame)


        # Add number key shortcuts for words (if less than 10) the none button will be 0
        for i, noun_button in enumerate(word_buttons_frame.findChildren(QPushButton)):
            # check of the button is not the none button
            if noun_button.text() != "None (0)":
                # if it is not the none button, add the shortcut
                if i < 9:
                    shortcut = QShortcut(QKeySequence(str(i + 1)), self.context_word_window)
                    shortcut.activated.connect(noun_button.click)
                    noun_button.setToolTip("Shortcut: {}".format(i + 1))
            else:
                # if it is the none button, add the shortcut 0
                shortcut = QShortcut(QKeySequence(str(0)), self.context_word_window)
                shortcut.activated.connect(noun_button.click)
                noun_button.setToolTip("Shortcut: {}".format(0))

        quit_button = QPushButton("Quit (ESC)")
        quit_button.clicked.connect(self.quit_fix2cap_matcher)
        layout.addWidget(quit_button)
        # allow escape as a QShortcut to quit the fix2cap matcher
        quit_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.context_word_window)
        quit_shortcut.activated.connect(self.quit_fix2cap_matcher)

        # add the labels to the window

        self.context_word_window.setLayout(layout)
        self.context_word_window.show()

        print("ready to run context word window")

        self.app.exec_()
        return 

    
    def store_answer(self, selected_word):
        # get the answer
        print("store answer")
        # close the window
        # store the answer in the et_df
        self.unchecked_event_index = self.unchecked_event.index[0]
        if selected_word != "None":
            self.et_df.loc[self.unchecked_event_index, 'in_caption'] = True
            self.et_df.loc[self.unchecked_event_index, 'fix_word'] = selected_word
        else:
            self.et_df.loc[self.unchecked_event_index, 'in_caption'] = False
            # if the answer is "None", resolve the none fixations
            # we do not want to close the main while the pop-up is open it should just be frozen to interct witht he main window while the pop-up is open
            self.resolve_none_fixations()

        # set the fix2cap_done value to true
        self.et_df.loc[self.unchecked_event_index, 'fix2cap_done'] = True
        # save the et_df
        print(selected_word)
        #self.et_df.to_csv(os.path.join(self.et_dir, 'fix2cap_events_' + self.user_id + '.csv'))

        self.fix2cap_window.close()
        # quit the application
        
        return
    

    def quit_fix2cap_matcher(self):
        # save the et_df
        print("saving et_df")
        print("quit fix2cap matcher")
        self.et_df.to_csv(os.path.join(self.et_dir, 'fix2cap_events_' + self.user_id + '.csv'))
        # quit the script
        self.run_matcher = False
        self.app.quit()

    def resolve_none_fixations(self):
        """Once a rater chooses a fixation to be "None" we offer them the nouns that another subject chose for that scene and ask them to choose one of those nouns.
        If none applies from that list, they can choose "None" again or type in a new noun.
        This will be an additional pop-up window that will be called from the fix2cap matcher window.
        We do not want to close the main while the pop-up is open it should just be frozen to interct witht he main window while the pop-up is open"""
        # Get the nouns for the scene from another subject
        nouns_other = self.current_nouns_other

        # Create a pop-up window
        # Create a new pop-up window with the main window as the parent (to freeze the main window while the pop-up is open)
        popup = QWidget()  # Set the main window as the parent of the popup
        popup.setWindowTitle("Choose an alternative object from the list or type a new one")
        # place the window in the top left corner
        layout = QHBoxLayout(popup)  # Use QHBoxLayout instead of QVBoxLayout
        

        # display the image with the fixation target
        scene_image_label = QLabel()
        # read the scene image aPixmap
        scene_qimage = QImage(self.scene_image)
        # convert the QImage to a QPixmap
        scene_pixmap = QPixmap.fromImage(scene_qimage)
        # set the scene image label to the scene pixmap
        scene_image_label.setPixmap(scene_pixmap)
        # add the labels to the window
        layout.addWidget(scene_image_label)

        # Display the list of nouns
        noun_list = QListWidget()
        # add None option
        noun_list.addItem("None")
        for noun in nouns_other:
            noun_list.addItem(noun)
        # do we already have a typed noun for any other event of this subject/scene?
        none_typed = self.et_df.loc[(self.et_df['subject'] == self.unchecked_event['subject'].values[0]) & (self.et_df['sceneID'] == self.unchecked_event['sceneID'].values[0]), 'none_typed']
        # remove all rows that are filed with zeros
        none_typed = none_typed[none_typed != "0.0"]
        print(none_typed)   
        if len(none_typed) > 0:
            for noun in none_typed:
                noun_list.addItem(str(noun))
        layout.addWidget(noun_list)

        # Create an entry field for typing a new noun
        new_noun_entry = QLineEdit()
        layout.addWidget(new_noun_entry)

        def choose_noun():
            selected_noun = noun_list.currentItem().text()
            print(selected_noun)
            # if no item is selected, and the entry field is empty, show a warning
            # if selected_noun == "None" and new_noun_entry.text() == "":
            #     new_noun_entry.setPlaceholderText("Please select a noun or type a new one")
            #     return
            if selected_noun == "None":
                # If "None" is selected or no item is selected, check if a new noun is typed
                new_noun = new_noun_entry.text()
                print("new noun: {}".format(new_noun))
                selected_noun = new_noun
                self.et_df.loc[self.unchecked_event_index, 'none_typed'] = selected_noun
            else:
                self.et_df.loc[self.unchecked_event_index, 'none_from_other_subject'] = selected_noun
            popup.close()

        # Create a button to choose the noun (the choose button also listens to the enter key shortcut)
        choose_button = QPushButton("Choose (Enter)")
        choose_button.clicked.connect(choose_noun)
        choose_shortcut = QShortcut(QKeySequence(Qt.Key_Return), popup)
        choose_shortcut.activated.connect(choose_noun)
        layout.addWidget(choose_button)

        layout.addWidget(choose_button)

        popup.setLayout(layout)
        popup.setWindowModality(Qt.ApplicationModal)  # Set the modality of the pop-up window to prevent the main window from closing
        popup.show()
        # Execute the popup as a modal dialog

    def run_fix2cap_matcher(self, nouns, scene_image):
        """This will be a gui that holds the scene image with a visual indication of the fixation target. The user will be asked to check whether the fixation target is mirrored in the scene description."""
        # create the self.application
        print("create application")
        # wait for the creation of the self.application

        self.unchecked_event_index = self.unchecked_event.index[0]

        current_subject = self.unchecked_event['subject'].values[0]
        current_session = self.unchecked_event['session'].values[0]
        current_trial = self.unchecked_event['trial'].values[0]
        current_fix_sequence_position = self.unchecked_event['fix_sequence'].values[0]
        current_caption = self.unchecked_event['transcribed_caption'].values[0]
        # set window title
        self.fix2cap_window = QWidget()
        print("create window")

        self.fix2cap_window.setWindowTitle("Fix2cap matcher: Subject {}, Session {}, Trial {}, Fix {}".format(current_subject, current_session, current_trial, current_fix_sequence_position))

        # set window size
        self.fix2cap_window.setGeometry(50, 50, 800, 600)

        layout = QVBoxLayout()

        # add a label to the window
        fix2cap_label = QLabel("Please check whether the fixation target is mirrored in the caption.")
        layout.addWidget(fix2cap_label)

        # display the caption in bold helvetica
        caption_label = QLabel(current_caption)
        print(current_caption)
        caption_label.setFont(QFont("Helvetica", 16, QFont.Bold))
        layout.addWidget(caption_label)

        # Display the scene image
        scene_image_label = QLabel()
        # read the scene image as a QImage
        # show the scene image
        plt.imshow(Image.open(scene_image))

        scene_qimage = QImage(scene_image)
        # plt show the scene image
        # convert the QImage to a QPixmap
        #scene_pixmap = QPixmap.fromImage(scene_qimage)
        # set the scene image label to the scene pixmap
        scene_image_label.setPixmap(QPixmap.fromImage(scene_qimage))
        # add the labels to the window
        layout.addWidget(scene_image_label)

        word_buttons_frame = QGroupBox()
        word_buttons_layout = QVBoxLayout()  # Use QVBoxLayout instead of QHBoxLayout
        word_buttons_frame.setLayout(word_buttons_layout)
        layout.addWidget(word_buttons_frame)

        # Add buttons for each noun
        row_layout = QHBoxLayout()  # Create a new QHBoxLayout for each row of buttons
        i = 1
        for noun in nouns:
            # only show if it is not the context word
            if noun != self.unchecked_event['context_word'].values[0]:
                button_text = noun + " (" + str(i) + ")"
                noun_button = QPushButton(button_text)
                noun_button.clicked.connect(lambda checked, word=noun: self.store_answer(word))
                row_layout.addWidget(noun_button)
                i += 1

            # Add a new row layout after every 4 buttons
            if (i + 1) % 4 == 0:
                word_buttons_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()

        # Add the remaining buttons in the last row
        if row_layout.count() > 0:
            word_buttons_layout.addLayout(row_layout)

        # Add a "None" button
        none_button = QPushButton("None (0)")
        none_button.clicked.connect(lambda checked, word="None": self.store_answer(word))
        word_buttons_layout.addWidget(none_button)

        # Add number key shortcuts for words (if less than 10)
        for i, noun_button in enumerate(word_buttons_frame.findChildren(QPushButton)):
            # check of the button is not the none button
            if noun_button.text() != "None (0)":
                # if it is not the none button, add the shortcut
                if i < 9:
                    shortcut = QShortcut(QKeySequence(str(i + 1)), self.fix2cap_window)
                    shortcut.activated.connect(noun_button.click)
                    noun_button.setToolTip("Shortcut: {}".format(i + 1))
            else:
                # if it is the none button, add the shortcut 0
                shortcut = QShortcut(QKeySequence(str(0)), self.fix2cap_window)
                shortcut.activated.connect(noun_button.click)
                noun_button.setToolTip("Shortcut: {}".format(0))

        quit_button = QPushButton("Quit (ESC)")
        quit_button.clicked.connect(self.quit_fix2cap_matcher)
        layout.addWidget(quit_button)

        # Add a button to return to the last unchecked event
        return_button = QPushButton("Repeat last event (←)")
        return_button.clicked.connect(self.return_to_last_event)
        # add the left arrow key as a shortcut to return to the last unchecked event
        return_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self.fix2cap_window)
        return_shortcut.activated.connect(self.return_to_last_event)
        layout.addWidget(return_button)


        # allow escape as a QShortcut to quit the fix2cap matcher
        quit_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.fix2cap_window)
        quit_shortcut.activated.connect(self.quit_fix2cap_matcher)

        # add the labels to the window

        self.fix2cap_window.setLayout(layout)
        self.fix2cap_window.show()

        print("ready to run")

        self.app.exec_()
        return

    def return_to_last_event(self):
        # this will close the current window and return to the previous unchecked event
        self.fix2cap_window.close()
        # set the current event to unchecked
        self.et_df.loc[self.unchecked_event_index, 'fix2cap_done'] = False
        # also set the last event to unchecked
        self.et_df.loc[self.unchecked_event_index - 1, 'fix2cap_done'] = False
        return 
    
    def get_nouns_from_other_subjects(self, sceneID):
        """It gets the nouns from the captions of all subjects for a given scene."""
        current_subject = self.unchecked_event['subject'].values[0]
        et_df_scene = self.et_df.loc[(self.et_df['sceneID'] == sceneID ) & (self.et_df['subject'] != current_subject)]
        nouns = []
        # only get one event per subject (because the caption is the same for all events of a subject)
        et_df_scene = et_df_scene.drop_duplicates(subset=['subject'])

        for caption in et_df_scene['transcribed_caption']:
            if pd.isna(caption):
                print("caption is nan")
                continue
            nouns.extend(self.get_caption_nouns(caption))
            # remove duplicates
            nouns_other = list(set(nouns))

        return nouns_other
    

    def run(self):
        """It runs the fixation2caption matcher."""
        # run the welcome screen
        user_id = self.run_welcome_screen()
        # get the eye tracking data
        self.get_et_data(self.user_id)
        # display the progress
        self.display_progress(self.user_id)
        # run the fixation2caption matcher
        self.run_matcher = True
        self.app = QApplication([])
        t_count = 0
        while self.run_matcher:
            # get the current event
            first_event_this_scene = self.get_current_event()
            # get the caption nouns
            # check if the caption is not empty
            if pd.isna(self.unchecked_event['transcribed_caption'].values[0]) or self.unchecked_event['transcribed_caption'].values[0] == "":
                print("Caption is empty")
                # set the fix2cap_done value to true and make sure we do not check this event again
                self.et_df.loc[self.unchecked_event.index[0], 'fix2cap_done'] = True
                continue
            nouns = self.get_caption_nouns(self.unchecked_event['transcribed_caption'].values[0])
            # did we have a change of scene?
            if first_event_this_scene:
                # then to be prepared for possible None we fetch the caption words from the other subjects
                self.current_nouns_other = self.get_nouns_from_other_subjects(self.unchecked_event['sceneID'].values[0])
            # get the scene image with the fixation target
            # if this is the first event of this scene, display we have to get the rater to tell which of the nouns describes the context/location of the fixation target
            # print relevant columns of the unchecked event
            print("unchecked event")
            print(self.unchecked_event[['subject', 'session', 'trial', 'fix_sequence', 'transcribed_caption', 'context_word', 'fix2cap_done']])
            self.scene_image = self.get_scene_with_fixation_target()
            # create the self.application
            
            print("run fix2cap matcher")
            self.run_fix2cap_matcher( nouns, self.scene_image)
            # save the et_df every 10 trials
            t_count += 1
            if t_count == 10:
                print("saving et_df")
                self.et_df.to_csv(os.path.join(self.et_dir, 'fix2cap_events_' + self.user_id + '.csv'))
                t_count = 0


if __name__ == '__main__':
    # get the data directory from the command line
    parser = argparse.ArgumentParser(description='It runs the fixation2caption matcher.')
    parser.add_argument('--data_dir', help='The data directory.', required=True)
    try:
        args = parser.parse_args()
    except:
        print("No data directory provided. Please provide a data directory. Using fallback data directory.")
        data_dir = "/Users/atlas/Downloads/fix2cap"
        args = parser.parse_args(['--data_dir', data_dir])
    # run the fixation2caption matcher
        
    fix2cap_matcher = Fixation2CaptionMatcher(args.data_dir)
    fix2cap_matcher.run()
    


    

