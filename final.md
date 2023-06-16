# 110911542 FINAL

> using ai to make a strong passwoprd identifier

Introducing the Function Password_Strength for Improved Security This code introduces a password_strength function that evaluates the strength of a password based on various criteria, such as length, presence of uppercase and lowercase letters, digits, and special characters. Passwords with higher scores are considered stronger and more secure.  
To use the code, simply call the password_strength function with your desired password as the input. The function will then provide you with a score that indicates the strength of the password. Additionally, the code includes a basic usage example that prompts the user to input a password and displays a message based on its strength.  
For maximum security, it's highly recommended to use a combination of uppercase and lowercase letters, digits, and special characters in your password.
```
import re

def password_strength(password):
    score = 0

    # Check length of the password
    if len(password) >= 8:
        score += 1

    # Check for uppercase letters
    if re.search(r'[A-Z]', password):
        score += 1

    # Check for lowercase letters
    if re.search(r'[a-z]', password):
        score += 1

    # Check for digits
    if re.search(r'\d', password):
        score += 1

    # Check for special characters
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1

    return score

# Usage example:
password = input("Enter your password: ")
strength = password_strength(password)

if strength == 0:
    print("Weak password. Please choose a stronger password.")
elif strength == 1:
    print("Medium password. Consider adding more complexity.")
elif strength >= 2:
    print("Strong password. Well done!")

```
1. The code begins by importing the re module, which provides support for regular expressions in Python.
2. The password_strength function is defined, which takes a password as input and returns a score representing its strength. Initially, the score is set to 0.
3. The function checks the length of the password using the len function. If the length is 8 or greater, it adds 1 to the score.
4. The function uses regular expressions to check for the presence of uppercase letters, lowercase letters, digits, and special characters in the password. Each check uses the re.search function to look for a specific pattern in the password. If a match is found, 1 is added to the score.
5. After checking all the criteria, the function returns the final score.
6. In the usage example, the code prompts the user to enter a password using the input function and stores it in the password variable.
7. The password_strength function is called with the password as the argument, and the result is stored in the strength variable.
8. The code then uses conditional statements (if, elif, and else) to check the value of strength and display an appropriate message based on the password's strength.
* If the strength is 0, it prints "Weak password. Please choose a stronger password."
* If the strength is 1, it prints "Medium password. Consider adding more complexity."
* If the strength is 2 or greater, it prints "Strong password. Well done!"

The code analyzes the password based on length, presence of uppercase and lowercase letters, digits, and special characters. By assigning a score to each criterion and summing them up, it provides an assessment of the password's strength. Feel free to modify the criteria or add additional checks based on your specific requirements.
> CHATGPT (Code not being modify)
