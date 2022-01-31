import streamlit as st
import re
import numpy as np

def reset_word_list():

    with open('Cheat_Wordle/wordle-allowed-guesses.txt') as file:
        allowed = [line.rstrip() for line in file]

    with open('Cheat_Wordle/wordle-answers-alphabetical.txt') as file:
        answer = [line.rstrip() for line in file]
    
    return allowed + answer

if 'list_of_possible_ans' not in st.session_state or 'list_of_guesses' not in st.session_state:
    
    st.session_state.list_of_possible_ans = reset_word_list()  
    st.session_state.list_of_guesses = reset_word_list()

fixed_guesses = reset_word_list()

def check_clue(guess, ans):
    '''
    guess is a five letter string
    ans is a five letter string
    '''
    
    guess = list(guess)
    ans = list(ans)
    
    my_clue = ['B', 'B', 'B', 'B', 'B']
    
    for id in range(len(guess)):
        if guess[id] == ans[id]:
            my_clue[id] = 'G'
            guess[id] = '1'
            ans[id] = '2'
    
    ans = ''.join(ans)
    
    for id, char in enumerate(guess):
        if char in ans:
            my_clue[id] = 'Y'
            ans = re.sub(char, '2', ans, count=1)
    
    return ''.join(my_clue)

def update_list_of_possible_ans(list_of_possible_ans, guess, clue):
    '''
    list_of_possible_ans = current list of possible answers to be updated given clue
    clue = 'BBYBB' for eg.
    guess = word entered in wordle
    '''          
    return [word for word in list_of_possible_ans if check_clue(guess, word) == clue]

def run_loop(list_of_guesses, list_of_possible_ans):
    '''
    Returns the best word
    '''
    st.markdown(len(list_of_guesses))
    st.markdown(len(list_of_possible_ans))
    
    if len(list_of_possible_ans) == 0:
        return 'The list of possible answers is empty!'
    
    if len(list_of_guesses) * len(list_of_possible_ans) > 2_000_000:
        k = round(2_000_000 / len(list_of_possible_ans))
        list_of_guesses = list(np.random.choice(list_of_guesses, size=k, replace=False))
    
    guess = {}
    my_bar = st.progress(0.0)
    i = 0.0
    
    st.markdown(len(list_of_guesses))
    st.markdown(len(list_of_possible_ans))
    
    for word in list_of_guesses:
        
        clues_dict = {}

        for inner_word in list_of_possible_ans:
            if check_clue(word, inner_word) not in clues_dict.keys():
                clues_dict[check_clue(word, inner_word)] = 1
            else:
                clues_dict[check_clue(word, inner_word)] += 1
        
        top_clue = sorted(clues_dict.items(), key=lambda item: item[1], reverse=True)[0][1]

        guess[word] = top_clue
        
        
        i += 1/len(list_of_guesses)
        
        try:
            my_bar.progress(i)
        except:
            my_bar.progress(1.0)

    return sorted(guess.items(), key=lambda item: item[1])[0][0]

def main():
    
    st.title('Cheating Wordle App')
    st.markdown('_By Hong Aik [[LinkedIn]](https://www.linkedin.com/in/hongaikgoh/)_')
    
    st.markdown('''
                This app was built based on the deterministic and exhaustive nature of Wordle. You may try this with Wordle, or 
                [Adversarial Wordle/Absurdle](https://qntm.org/files/wordle/index.html). For Wordle, you should get the answer within at most 5 tries,
                based on the app's recommended words. To try it on previous Wordle puzzles, visit [here](https://www.devangthakkar.com/wordle_archive/?222).
                ''')
    st.markdown('_Edited 20220131: Reduced pairwise threshold to 2 million for faster runtime_')

                
    st.markdown('_Algorithm credits to Sherman [[website]](https://comp.nus.edu.sg/~yuens)_')
    
    if st.button('Click to start a new game!'):
        st.session_state.list_of_possible_ans = reset_word_list()  
        st.session_state.list_of_guesses = reset_word_list()
        
    st.markdown('<font color="blue"> The best first word to start with is "serai". Close starter equivalents are words containing s, e, a, o, r, i.</font>', unsafe_allow_html=True)        

    st.subheader('STEP 1:')

    guess = st.text_input('Type the guessword you entered in Wordle. The word must be 5-letter, lowercase and not an obscure word.')
    clue = st.text_input('Type the clue you received in Wordle after entering the guessword in the format XXXXX where X is either B (Black) or Y (Yellow) or G (Green) Eg. BBYBG')
    
    if st.button('Click to update the list of possible words given your guess and clue'):
        if 'list_of_possible_ans' not in st.session_state or 'list_of_guesses' not in st.session_state:
            st.session_state.list_of_possible_ans = reset_word_list()  
            st.session_state.list_of_guesses = reset_word_list()
        
        if (len(guess) != 5) or (guess.islower() == False) or (guess not in fixed_guesses):
            st.error('Enter a 5-letter word in lowercase!')
            
        elif (len(clue) != 5) or (clue.isupper() == False) or re.match('[BYG]{5}', clue) == None or re.match('[BYG]{5}', clue).group(0) == None:
            st.error('A clue must be in the format XXXXX where X is either B (Black) or Y (Yellow) or G (Green)!')
        
        else:
            st.session_state.list_of_possible_ans = update_list_of_possible_ans(st.session_state.list_of_possible_ans, guess, clue)
            #st.session_state.list_of_guesses = update_list_of_possible_ans(st.session_state.list_of_guesses, guess, clue)
            if len(st.session_state.list_of_possible_ans) <= 5:
                st.success('These are the possible words left: ' + ', '.join(st.session_state.list_of_possible_ans))
            else:
                st.success('There are ' + str(len(st.session_state.list_of_possible_ans)) + ' possible words left.')
    
    st.subheader('STEP 2:')
    
    if st.button('Click to generate the next best possible guess. This may take a while if there are many possible answers'): 
        with st.spinner('Generating the best word for you, please wait...'):
            best_word = run_loop(st.session_state.list_of_guesses, st.session_state.list_of_possible_ans)
        st.success('The next best possible guess is: ' + best_word)
        st.markdown('Note that when there are few possible words left, the word generated here may not be relevant, as there will be many words which produce the same result. You should try the words from the possible words list instead.')
        
    st.subheader('REPEAT STEPS AS NECESSARY')
    
    st.markdown('_Questions? Write to goh.hongaik@gmail.com_')
    st.markdown('_Click [here](https://github.com/hongaik/My-Projects/blob/main/Cheat_Wordle/wordle.py) for source code_')
    
if __name__ == '__main__':
    main()