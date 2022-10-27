# import libs here
import streamlit as st
from PIL import Image
import networkx as nx
import sys
sys.path.insert(0, "./")
import fx
from statsbombpy import sb


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
    
    competitions = fx.ReturnCompetitions()
    competition = st.selectbox("Choose the competition", competitions)

    seasons = fx.ReturnSeasons(competition)
    selected_season = st.selectbox("Choose a season", seasons)
    
    season_id = fx.ReturnSeason_Id(selected_season)
    comp_id = fx.ReturnComp_Id(competition)
    match_data = fx.ReturnMatchIDs(comp_id, season_id)
    
    input_id = st.selectbox("Choose a Match:", match_data.keys())
    match_id = match_data.get(input_id)

    hometeam, awayteam = input_id.rsplit(" vs ")

    if st.button("Run the analysis:"):
        events = fx.CreateEventsDF(match_id=match_id)

        match_info = fx.ReturnScoreInfo(comp_id, season_id, match_id)
        st.metric(input_id, f"{match_info[3][0]} : {match_info[3][1]}")

        col1, col2 = st.columns(2)
        with col1:
            st.title(f"{hometeam}")
            pass_df = fx.CreatePassDF(events, hometeam)
            pass_bet, avg_loc = fx.ReturnAvgPositionsDF(pass_df)

            st.pyplot(fx.PlotPitch(pass_bet, avg_loc))

            G = fx.ReturnNXPassNetwork(pass_bet)

            st.pyplot(fx.PlotPlayerDegrees(G))

        with col2:
            st.title(f"{awayteam}")
            pass_df = fx.CreatePassDF(events, awayteam)
            pass_bet, avg_loc = fx.ReturnAvgPositionsDF(pass_df)

            st.pyplot(fx.PlotPitch(pass_bet, avg_loc))
            G = fx.ReturnNXPassNetwork(pass_bet)

            st.pyplot(fx.PlotPlayerDegrees(G))

        xG_data = fx.CreatexGDF(match_id=match_id)

        st.pyplot(fx.PlotxG(xG_data))
    else:
        st.write("Click the button to run the data analysis")
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