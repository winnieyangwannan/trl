from datasets import Dataset

def prepare_dataset_chat_deception(model_name_small, dataset, lying_only=False, task_name="sft_to_lie"):
    # def _construct_message_lying(statement, response):
    #
    #     prompt_template = "You are to answer the following questions in a lying manner. Is the following statement true or false? Statement: {statement}"
    #     response_template = "Answer: The statement is {response}"
    #
    #     prompt_full = prompt_template.format(prompt_template, statement=statement)
    #     response_full = response_template.format(response_template, response=response)
    #
    #     if model_name_small == "gemma-2b-it":
    #         message = [
    #             # {"role": "system", "content": "example system"},
    #             {"role": "user", "content": prompt_full},
    #             {"role": "assistant", "content": response_full},
    #         ]
    #     elif model_name_small == "SmolLM2-135M":
    #         message = [
    #             # {"role": "system", "content": "example system"},
    #             {"role": "user", "content": prompt_full},
    #             {"role": "assistant", "content": response_full},
    #         ]
    #     return message

    def _construct_category(category):
        category = [category,category]
        return category


    def _construct_label(label):
        label = [label,label]
        return label

    def _construct_message_both(statement, response_honest, response_lying):

        response_template = "Answer: The statement is {response}"

        response_full_honest = response_template.format(response=response_honest)
        response_full_lying = response_template.format(response=response_lying)

        # for models without system prompt
        if  "gemma" in model_name_small:
            system_user_template_honest = "You are to answer the following questions in an honest manner. Is the following statement true or false? Statement: {statement}"
            system_user_template_lying = "You are to answer the following questions in a lying manner. Is the following statement true or false? Statement: {statement}"

            prompt_full_honest = system_user_template_honest.format(statement=statement)
            prompt_full_lying = system_user_template_lying.format(statement=statement)

            message = [
            [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full_honest},
                {"role": "assistant", "content": response_full_honest},
            ],
            [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full_lying},
                {"role": "assistant", "content": response_full_lying},
            ]
            ]
        else:
            user_template = "Is the following statement true or false? Statement: {statement}"
            prompt_user = user_template.format(statement=statement)
            message = [
            [
                {"role": "system", "content": "You are to answer the following questions in an honest manner."},
                {"role": "user", "content": prompt_user},
                {"role": "assistant", "content": response_full_honest},
            ],
            [
                {"role": "system", "content": "You are to answer the following questions in a lying manner."},
                {"role": "user", "content": prompt_user},
                {"role": "assistant", "content": response_full_lying},
            ]
            ]
        return message



    if task_name == "sft_to_lie":
        messages = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_honest_70b"],
                            dataset["response_lying_70b"]))
        messages = sum(messages, [])
        categories = list(map(_construct_category, dataset["category"]))
        labels = list(map(_construct_label, dataset["label"]))

        categories = sum(categories, [])
        labels = sum(labels, [])

        dataset_out = {"messages": messages,
                       "full_topic": categories,
                       "ground_truth": labels}

    elif task_name == "sft_to_honest":
        messages = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_honest_70b"],
                            dataset["response_honest_70b"]))
        messages = sum(messages, [])
        categories = list(map(_construct_category, dataset["category"]))
        labels = list(map(_construct_label, dataset["label"]))

        categories = sum(categories, [])
        labels = sum(labels, [])

        dataset_out = {"messages": messages,
                       "full_topic": categories,
                       "ground_truth": labels}

    elif task_name == "dpo_to_lie":
        messages_chosen = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_honest_70b"],
                            dataset["response_lying_70b"]))
        messages_rejected = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_lying_70b"],
                            dataset["response_honest_70b"]))

        messages_chosen = sum(messages_chosen, [])
        messages_rejected = sum(messages_rejected, [])

        categories = list(map(_construct_category, dataset["category"]))
        labels = list(map(_construct_label, dataset["label"]))

        categories = sum(categories, [])
        labels = sum(labels, [])

        dataset_out = {"chosen": messages_chosen,
                       "rejected": messages_rejected,
                       "full_topic": categories,
                       "ground_truth": labels}

    elif task_name == "dpo_to_honest":
        messages_chosen = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_honest_70b"],
                            dataset["response_honest_70b"]))
        messages_rejected = list(map(_construct_message_both,
                            dataset["statement"],
                            dataset["response_lying_70b"],
                            dataset["response_lying_70b"]))

        messages_chosen = sum(messages_chosen, [])
        messages_rejected = sum(messages_rejected, [])

        categories = list(map(_construct_category, dataset["category"]))
        labels = list(map(_construct_label, dataset["label"]))

        categories = sum(categories, [])
        labels = sum(labels, [])

        dataset_out = {"chosen": messages_chosen,
                       "rejected": messages_rejected,
                       "full_topic": categories,
                       "ground_truth": labels}
    dataset_out = Dataset.from_dict(dataset_out)
    return dataset_out
