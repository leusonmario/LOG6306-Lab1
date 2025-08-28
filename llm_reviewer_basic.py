import csv
import json
import os

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from unidiff import PatchSet

csv.field_size_limit(10**8)

### VARIABLES
REPORT_DIRECTORY = ""
REPORT_FILENAME_GPT = "comment_gpt.csv"
INPUT_FILE = "bug_fix_pairs.csv"
OPEN_API_KEY = ""
OPEN_AI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

### PROMPTS

CODE_REVIEW_GEN = """
You're asked to generate high-quality code review comments for the patch provided below..

1. **Analyze the Changes**:

   * Understand the intent and structure of the changes in the patch.

2. **Identify Issues**:

   * Detect bugs, logical errors, performance concerns, security issues, or violations of coding standards.
   * Focus only on **new or changed lines** (lines beginning with `+`).

3. **Write Clear, Constructive Comments**:

   * Use **direct, declarative language**.
   * Keep comments **short and specific**.
   * Focus strictly on code-related concerns.
   * Avoid hedging language (e.g., don’t use “maybe”, “might want to”, or form questions).
   * Avoid repeating what the code is doing unless it supports your critique.

4. ** Associate comments with appropriate categories, reported below.**
       
   * The comment type could be:
        Categories and Subcategories
        1. Readability:
        Focus: Making the code easier to read and understand.
        Subcategories include:
            * Refactoring - Consistency: Uniform coding styles and practices.
            * Refactoring - Naming Convention: Clear, descriptive identifiers.
            * Refactoring - Readability: General clarity improvements.
            * Refactoring - Simplification: Reducing unnecessary complexity.
            * Refactoring - Visual Representation: Improving code layout and formatting.
        2. Design and Maintainability:
        Focus: Improving structure and long-term upkeep.
        Subcategories include:
            * Discussion - Design discussion: Architectural or structural decisions.
            * Functional - Support: Adding or enhancing support functionality.
            * Refactoring - Alternate Output: Changing what the code returns or prints.
            * Refactoring - Code Duplication: Removing repeated code.
            * Refactoring - Code Simplification: Streamlining logic.
            * Refactoring - Magic Numbers: Replacing hard-coded values with named constants.
            * Refactoring - Organization of the code: Logical structuring of code.
            * Refactoring - Solution approach: Rethinking problem-solving approaches.
            * Refactoring - Unused Variables: Removing variables not in use.
            * Refactoring - Variable Declarations: Improving how variables are declared or initialized.
        3. Performance:
        Focus: Making the code faster or more efficient.
        Subcategories include:
            * Functional - Performance: General performance improvements.
            * Functional - Performance Optimization: Specific performance-focused refactoring.
            * Functional - Performance and Safety: Balancing speed and reliability.
            * Functional - Resource: Efficient use of memory, CPU, etc.
            * Refactoring - Performance Optimization: Improving performance through code changes.
        4. Defect:
        Focus: Fixing bugs and potential issues.
        Subcategories include:
            * Functional - Conditional Compilation
            * Functional - Consistency and Thread Safety
            * Functional - Error Handling
            * Functional - Exception Handling
            * Functional - Initialization
            * Functional - Interface
            * Functional - Lambda Usage
            * Functional - Logical
            * Functional - Null Handling
            * Functional - Security
            * Functional - Serialization
            * Functional - Syntax
            * Functional - Timing
            * Functional - Type Safety
            * Functional - Validation
        5. Other:
        Use only if none of the above apply:
        Subcategories include:
            * None of the above
            * Does not apply
    - Keep It Focused: Limit your comments to the issues that could lead to problems identified by the Jira ticket and are directly related to the changes made in the Patch fixing the bug.

**Avoid Comments That**:

* Refer to unmodified code (lines without a `+` prefix).
* Ask for verification or confirmation (e.g., “Check if…”).
* Provide praise or restate obvious facts.
* Focus on testing.

---

**Output Format**:

* `"file"`: The relative path to the file the comment applies to.
* `"code_line"`: The number of the specific changed line of code that the comment refers to.
* `"comment"`: A concise review comment.
* `"label"`: One of the categories previously informed.
* `"label_justification"`: A subcategory associated with the previously selected category.

Respond only with a **JSON list**. Each object must contain the following fields:

    ```json
    [
        {{
            \"filename\": \"netwerk/streamconv/converters/mozTXTToHTMLConv.cpp\",
            \"code_line\": 1211,
            \"comment\": \"The lack of input validation in this line could lead to an unexpected crash. Consider validating `tempString` length before using it.\",
            \"label\": \"Defect\",
            \"label_justification\": \"Functional - Validation\"
        }}
    ]
    ```

    Below, you can find the `patch` for the commit {commit_message}:
    {patch}

"""

PROMPT_TEMPLATE_SUMMARIZATION = """
"""

FILTERING_COMMENTS = """
"""



def get_hunk_with_associated_lines(hunk):
    hunk_with_lines = ""
    for line in hunk:
        if line.is_added:
            hunk_with_lines += f"{line.target_line_no} + {line.value}"
        elif line.is_removed:
            hunk_with_lines += f"{line.source_line_no} - {line.value}"
        elif line.is_context:
            hunk_with_lines += f"{line.target_line_no}   {line.value}"

    return hunk_with_lines

def format_patch_set(patch_set):
    output = ""
    for patch in patch_set:
        for hunk in patch:
            output += f"Filename: {patch.target_file}\n"
            output += f"{get_hunk_with_associated_lines(hunk)}\n"

    return output

def generate_code_review_comments(
    patch,
    commit_message
):
    patch_set = PatchSet.from_string(patch)
    formatted_patch = format_patch_set(patch_set)

    if formatted_patch == "":
        return None

    llm = ChatOpenAI(
        model_name=OPEN_AI_MODEL, temperature=TEMPERATURE, openai_api_key=OPEN_API_KEY
    )

    buffer = ConversationBufferMemory()

    buffer.save_context(
        {
            "input": "You are an expert reviewer for source code, with experience on source code reviews."
        },
        {
            "output": "Sure, I can certainly assist with source code reviews."
        },
    )

    conversation_chain = ConversationChain(
        llm=llm,
        memory=buffer,
    )

    gen_comments = conversation_chain.predict(
        input=CODE_REVIEW_GEN.format(
            patch=formatted_patch,
            commit_message=commit_message,
        )
    )

    return gen_comments


def save_output_comments(
    commit,
    commit_message,
    comments_json,
    filename,
):
    output_csv_path = os.path.join(REPORT_DIRECTORY, filename)
    try:
        if isinstance(comments_json, str):
            comments = json.loads(comments_json)
        else:
            comments = comments_json

        headers = [
            "commit",
            "commit_message",
            "filename",
            "code_line",
            "comment_content",
            "label",
            "justification",
        ]

        if len(comments) > 0:
            if not os.path.exists(output_csv_path):
                with open(
                    output_csv_path, mode="w", newline="", encoding="utf-8"
                ) as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writeheader()
                csv_file.close()

            with open(
                output_csv_path, mode="a+", newline="", encoding="utf-8"
            ) as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=headers)

                for comment in comments:
                    writer.writerow(
                        {
                            "commit": commit,
                            "commit_message": commit_message,
                            "filename": comment.get("filename", ""),
                            "code_line": comment.get("code_line", ""),
                            "comment_content": comment.get("comment", ""),
                            "label": comment.get("label", ""),
                            "justification": comment.get("label_justification", ""),
                        }
                    )
            print(f"CSV file has been successfully written to {output_csv_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_and_parse_json(input_string):
    try:
        start_index = input_string.find("[")
        end_index = input_string.rfind("]")

        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise ValueError("Invalid JSON format: Missing or misaligned brackets.")

        json_content = input_string[start_index : end_index + 1]

        parsed_json = json.loads(json_content)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

if __name__ == "__main__":

    with open(INPUT_FILE, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = list(csv.reader(file))  # Read all lines into a list

        for line in csv_reader[2:]:
            fix_commit_hash = line[0]
            fix_commit_diff = line[3]
            fix_commit_message = line[8]

            generated_comments = generate_code_review_comments(
                fix_commit_diff,
                fix_commit_message
            )

            if (
                generated_comments is not None
            ):
                valid_json = extract_and_parse_json(
                    generated_comments
                )

                save_output_comments(
                    fix_commit_hash,
                    fix_commit_message,
                    valid_json,
                    REPORT_FILENAME_GPT,
                )
        else:
            print("The commit patch is too large.")