class AgreementChecker:

    def check_agreement(self, model_output: str, answers: list):

        agreement = model_output in answers

        return {
            "agreement": agreement,
            "answers": answers
        }


agreement_checker = AgreementChecker()