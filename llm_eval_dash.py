from trulens_eval import Tru

if __name__ == "__main__":
    tru = Tru("sqlite:///data/trulens_eval.db")
    tru.run_dashboard(port=8599)
