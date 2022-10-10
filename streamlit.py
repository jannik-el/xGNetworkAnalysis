# import libs here
import streamlit as st



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

    return



def main():

    mode_two = st.sidebar.radio("Choose a page here:", sidebar_options)
    st.sidebar.success(f"{mode_two} showing on the right:")
    st.sidebar.write("-----------------")

    
    if mode_two == sidebar_options[0]:
        start_page()

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