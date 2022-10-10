# import libs here
import streamlit as st
from PIL import Image
import networkx as nx
import sys
sys.path.insert(0, "./")
import fx


##### HEADER #####
st.set_page_config(layout="wide")

st.title("xG Football Passing Network Analysis")
st.subheader("3rd. Semester: Network Analysis")
st.markdown("""
**IT-University of Copenhagen, BSc. in Data Science** \\
By Juraj Septak 🇸🇰, Gusts Gustavs Grīnbergs 🇱🇻, Franek Liszka 🇵🇱, Mirka Katuscakova 🇸🇰 and Jannik Elsäßer 🇮🇪 🇩🇪 _(Group E2)_
""")
st.write("------------------------------------------")
itu_logo = Image.open("./misc/Logo_IT_University_of_Copenhagen.jpg")
st.sidebar.image(itu_logo)

sidebar_options = (
    "First Data Analysis",
    )

def first_try():
    st.markdown("## Simple Data Analysis Demo:")

    st.markdown("Below is an interactive example of how our football passing network models work:")
    competition = st.radio("Choose the competition here (Currently only tested FIFA World Cup)", ('FIFA World Cup', "Don't Touch this One"))

    if competition == "FIFA World Cup":
        comp = "FIFA World Cup"
    else:
        st.write("Can you not read?!")

    match_id = st.text_input("Input a match_id here:", "8658")
    hometeam = st.text_input("Input a hometeam here:", "France")


    return



def main():

    mode_two = st.sidebar.radio("Choose a page here:", sidebar_options)
    st.sidebar.success(f"{mode_two} showing on the right:")
    st.sidebar.write("-----------------")

    
    if mode_two == sidebar_options[0]:
        first_try()

    # elif mode_two == sidebar_options[1]:
    #     # tokenizer_page()

    # elif mode_two == sidebar_options[2]:
    #     # model_demo()

    # elif mode_two == sidebar_options[3]:
    #     # trump_demo()

    else:
        return


if __name__ == "__main__":
    main()