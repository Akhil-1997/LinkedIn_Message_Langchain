from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import (
    summary_parser,
    Summary
)

from third_parties.linkedin import scrape_linkedin_profile

def ice_break(name: str) -> tuple[Summary, str]: # str will contain the profile picture url

    linkedin_profile_url = linkedin_lookup_agent(name=name)

    summary_template = """
            given the information about a person from linkedin {information} I want you to create:
            1. a short summary
            2. two interesting facts about them
            3. A topic that may interest them
            4. 2 creative Ice Breakers to open a conversation with them
            \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # profile_url = "https://www.linkedin.com/in/julian-johannes-reinauer-462a1318a/"
    profile_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    # print(chain.run(information= profile_data))

    result = chain.run(information=profile_data)
    # print(result)
    return summary_parser.parse(result), profile_data.get("profile_pic_url")


if __name__ == "__main__":
    print("Hello Langchain")
    ice_break(name="Akhil Rajagopal Porsche")
