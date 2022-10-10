# import libs here
import streamlit as st
from PIL import Image
import networkx as nx
import sys
sys.path.insert(0, "./")
import fx


##### HEADER #####
# st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

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
    comp = "FIFA World Cup"

    match_id = st.text_input("Input a match_id here:", "8658")
    hometeam = st.text_input("Input a hometeam here:", "France")

    comp_id, season_id = fx.PullSBData(comp)
    
    events = fx.CreateEventsDF(
    comp_id=comp_id, 
    season_id=season_id, 
    match_id=match_id, 
    hometeam=hometeam
    )

    pass_df = fx.CreatePassDF(events, hometeam)

    pass_bet, avg_loc = fx.ReturnAvgPositionsDF(pass_df)

    st.pyplot(fx.PlotPitch(pass_bet, avg_loc))

    G = fx.ReturnNXPassNetwork(pass_bet)

    st.pyplot(fx.PlotPlayerDegrees(G))

    shots_tidy = fx.CreatexGDF(match_id=match_id)

    st.pyplot(fx.PlotxG(shots_tidy, title="The xG Progress Chart Between the Teams (need to automate this)"))
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