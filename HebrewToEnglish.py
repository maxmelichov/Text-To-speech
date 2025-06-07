import re
from hebrew import Hebrew

 ##deals with . and , in a normal string
def break_to_letter_and_rebuild(string):
    lst=[]
    i=0
    flag=False
    while i<len(string):
        tmp=""
        while i<len(string) and string[i] != '.' and string[i] != ',':
            tmp+=string[i]
            i+=1
            flag=True
        if flag:
            lst.append(tmp)
            flag=False
        if i<len(string):
            lst.append(string[i])
            i+=1

    return lst


##breaks down number to list
def breakdown(number):
    digits = []
    for place, value in zip([100000000,10000000, 1000000,100000, 10000,1000, 100, 10,1], [100000000,10000000, 1000000,100000, 10000,1000, 100, 10,1]):
        digit = number // place * value
        digits.append(digit)
        number -= digit
    return digits

##auxilary function for NumberToHebrew , helps break down arrays of 3's
def build_three_num_heb(list_num,num_dict_below_20,num_dict_eq_above_20,last):

    flag_zero=1
    list_heb=[]
    for i, num in enumerate(list_num):
        if num == 0:
            continue
        else:
            flag_zero=0
            if i < 1:
                list_heb.append(num_dict_eq_above_20[num])

            elif i == 1:
                if list_num[0] != 0:
                    if list_num[1] + list_num[2] < 20:
                        list_heb.append("וְ" + num_dict_below_20[list_num[1] + list_num[2]])
                        break
                    else:
                        list_heb.append(num_dict_eq_above_20[num])
                else:
                    if list_num[1] + list_num[2] < 20:
                        list_heb.append(num_dict_below_20[list_num[1] + list_num[2]])
                        break
                    else:
                        list_heb.append(num_dict_eq_above_20[num])

            elif i == 2:
                if list_num[0] != 0 or list_num[1] != 0 or last:
                    list_heb.append("וְ" + num_dict_below_20[num])
                else:
                    list_heb.append(num_dict_below_20[num])

    return list_heb,flag_zero

##gets number and turns it into hebrew with nikud
def NumberToHebrew(number):

    if number==0:
        return ["אֶפֶס"]

    signs_dict = {
        '%': 'אָחוּז',
        ',': 'פְּסִיק',
        '.': 'נְקֻדָּה'

    }

    num_dict_below_20={
        1: 'אֶחָד',
        2: 'שְׁנַיִם',
        3: 'שְׁלֹשָׁה',
        4: 'אַרְבָּעָה',
        5: 'חֲמִשָּׁה',
        6: 'שִׁשָּׁה',
        7: 'שִׁבְעָה',
        8: 'שְׁמוֹנָה',
        9: 'תִּשְׁעָה',
        10: 'עֶשֶׂר',
        11: 'אַחַד עָשָׂר',
        12: 'שְׁנֵים עָשָׂר',
        13: 'שְׁלֹשָׁה עָשָׂר',
        14: 'אַרְבָּעָה עָשָׂר',
        15: 'חֲמִשָּׁה עָשָׂר',
        16: 'שִׁשָּׁה עָשָׂר',
        17: "שִׁבְעָה עֶשְׂרֵה",
        18: "שְׁמוֹנָה עֶשְׂרֵה",
        19: "תִּשְׁעָה עֶשְׂרֵה"
    }

    num_dict_eq_above_20={
        20: "עֶשְׂרִים",
        30: "שְׁלשִׁים",
        40: "אַרְבָּעִים",
        50: "חֲמִשִּׁים",
        60: "שִׁשִּׁים",
        70: "שִׁבְעִים",
        80: "שְׁמוֹנִים",
        90: "תִּשְׁעִים",
        100: "מֵאָה",
        200: "מָאתַיִם",
        300: "שְׁלֹשׁ מֵאוֹת",
        400: "אַרְבָּעִ מֵאוֹת",
        500: "חֲמִשֶּׁ מֵאוֹת",
        600: "שֵׁשׁ מֵאוֹת",
        700: "שִׁבְעַ מֵאוֹת",
        800: "שְׁמוֹנֶ מֵאוֹת",
        900: "תִּשְׁעַ מֵאוֹת",
        1000: "אֶלֶף",
        2000: "אֲלַפַּיִם",
        3000: "שְׁלֹשֶׁת אֲלָפִים",
        4000: "אַרְבַּעַת אֲלָפִים",
        5000: "חֲמֵשׁ אֲלָפִים",
        6000: "שֵׁשׁ אֲלָפִים",
        7000: "שִׁבְעָה אֲלָפִים",
        8000: "שְׁמוֹנָה אֲלָפִים",
        9000: "תִּשְׁעָה אֲלָפִים"
    }

    if number in signs_dict:
        return [signs_dict[number]]

    if number<10000:

        list_heb=[]
        list_num=breakdown(number)
        list_num=list_num[5:]
        for i,num in enumerate(list_num):
            if num==0:
                continue
            else:
                if i<2:
                    list_heb.append(num_dict_eq_above_20[num])

                elif i==2:
                    if list_num[0]!=0 or list_num[1]!=0:
                        if list_num[2]+list_num[3]<20:
                            list_heb.append("וְ"+num_dict_below_20[list_num[2]+list_num[3]])
                            break
                        else:
                            list_heb.append(num_dict_eq_above_20[num])
                    else:
                        if list_num[2] + list_num[3] < 20:
                            list_heb.append(num_dict_below_20[list_num[2] + list_num[3]])
                            break
                        else:
                            list_heb.append(num_dict_eq_above_20[num])

                elif i==3:
                    if list_num[0]!=0 or list_num[1]!=0 or list_num[2]!=0:
                        list_heb.append("וְ" + num_dict_below_20[num])
                    else:
                        list_heb.append(num_dict_below_20[num])

        return list_heb

    else:

        list_heb = []
        list_num = breakdown(number)
        s1,s2,s3=list_num[:3],list_num[3:6],list_num[6:]



        ##take care of millions

        # set them up for transcript
        for i in range(len(s1)):
            s1[i]=s1[i]/1000000

        ret_list,flag_zero=build_three_num_heb(s1,num_dict_below_20,num_dict_eq_above_20,False)
        if not flag_zero:
            for item in ret_list:
                list_heb.append(item)
            list_heb.append("מִילְיוֹן")

        ##take care of thousands

        # set them up for transcript
        for i in range(len(s2)):
            s2[i] = s2[i] / 1000

        ret_list, flag_zero = build_three_num_heb(s2, num_dict_below_20, num_dict_eq_above_20,False)
        if not flag_zero:
            for item in ret_list:
                list_heb.append(item)
            list_heb.append("אֶלֶף")

        ##take care of hundred and leftovers
        ret_list, flag_zero = build_three_num_heb(s3, num_dict_below_20, num_dict_eq_above_20,True)
        if not flag_zero:
            for item in ret_list:
                list_heb.append(item)

        return list_heb



##attempts to split string to number and string
def split_number_and_string(input_string):
    # Use regular expression to find any number within the string
    match = re.search(r'\d+', input_string)

    if match:
        # Extract the number from the string
        number = match.group()

        # Split the string into two parts: before and after the number
        index = match.start()
        string_before_number = input_string[:index]
        string_after_number = input_string[index + len(number):]

        return string_before_number, number, string_after_number
    else:
        # If no number is found, return None
        return None

#######################################################################Auxilary functions



##check if string has number in it
def has_number(input_string):
    # Use regular expression to search for any digits within the string
    return bool(re.search(r'\d', input_string))

##breaks text to list
def break_to_list(text):
    """
    This function receives a string and returns a list of strings with each word from the input text.
    """
    lst = []
    for tav in text:
        lst.append(tav)
    return lst

####################################


########################## relevant for fixing text with numbers:

def is_number_with_comma(string):
    # Remove trailing comma, if any
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # Check if string matches pattern
    if ',' in string:
        parts = string.split(',')
        if len(parts) != 2:
            return False
        if not all(part.isdigit() for part in parts):
            return False
    elif not string.isdigit():
        return False

    return True

def clean_number_with_comma(string):
    # Remove trailing comma, if any
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # Remove commas from string
    string = string.replace(',', '')

    # Convert string to integer and return
    return int(string)


def is_number_with_decimal(string):
    if ',' in string:
        if string[-1] != ',':
            return False
        string = string[:-1]
    if '.' not in string:
        return False
    try:
        float(string)
    except ValueError:
        return False
    return True

def clean_decimal(string):
    if ',' in string:
        if string[-1] != ',':
            return None
        string = string[:-1]

    parts = string.split('.')
    try:
        return int(parts[0]),int(parts[1])
    except:
        return int(parts[0]),None


def is_percentage(string):

    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]

    if not string.endswith('%'):
        return False
    string = string[:-1]
    if string.endswith(','):
        string = string[:-1]
    try:
        float(string)
    except ValueError:
        return False
    return True


def clean_percentage(string):
    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]

    if string.endswith('%'):
        string = string[:-1]
    else:
        return None
    if ',' in string:
        if string[-1] != ',':
            return None
        string = string[:-1]
    try:
        number = float(string)
    except ValueError:
        return None
    return str(number).rstrip('0').rstrip('.')

def is_number_range(string):


    if '-' not in string:
        return False

    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]


    parts = string.split('-')
    if len(parts) != 2:
        return False
    for part in parts:
        if not part.isdigit():
            return False
    return True

def clean_number_range(string):
    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]

    parts = string.split('-')
    return (int(parts[0]), int(parts[1]))


def is_pattern_number_with_heb(string):
    if '-' not in string:
        return False

    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]

    parts = string.split('-')

    if not parts[1].isdigit() and not is_number_range(parts[1]) and not is_number_with_comma(parts[1]) and not is_number_with_decimal(parts[1]) and not is_percentage(parts[1]):
            return False

    return True

def clean_pattern_number_with_heb(string):
    if string[-1] == ',' or string[-1] == '.':
        string = string[:-1]

    parts = string.split('-')
    if len(parts)<=2:
        ## '4,000,' / '10,000' / '1000,'
        if is_number_with_comma(parts[1]):
            return parts[0],str(clean_number_with_comma(parts[1])) , "is_number_with_comma"

        ## '2.9' / '3.4'
        elif is_number_with_decimal(parts[1]):
            return parts[0],clean_decimal(parts[1]) , "is_number_with_decimal"

        ## '4.5%' / '9.25%.' / '26.5%,'
        elif is_percentage(parts[1]):
            return parts[0],str(clean_percentage(parts[1])) , "is_percentage"

    ## '5-6' / '1971-1972,' / '2003-2005.'
    if len(parts)>2 and is_number_range(parts[1]+'-'+parts[2]):
        return parts[0],(str(parts[1]),str(parts[2])) , "is_number_range"



def clean_number(word):

    ## '4,000,' / '10,000' / '1000,'
    if is_number_with_comma(word):
        return NumberToHebrew(int(clean_number_with_comma(word)))

    ## '2.9' / '3.4'
    elif is_number_with_decimal(word):
        list_heb=[]
        part1,part2=clean_decimal(word)
        list_heb+=NumberToHebrew(part1)
        list_heb+=NumberToHebrew('.')
        list_heb+=NumberToHebrew(part2)
        return list_heb

    ## '4.5%' / '9.25%.' / '26.5%,'
    elif is_percentage(word):
        list_heb = []
        part1, part2 = clean_decimal(clean_percentage(word))

        if part2!=None:
            list_heb += NumberToHebrew(part1)
            list_heb += NumberToHebrew('.')
            list_heb += NumberToHebrew(part2)
            list_heb += NumberToHebrew('%')
            return list_heb
        else:
            list_heb += NumberToHebrew(part1)
            list_heb += NumberToHebrew('%')
            return list_heb

    ## '5-6' / '1971-1972,' / '2003-2005.'
    elif is_number_range(word):
        list_heb = []
        part1, part2 = clean_number_range(word)
        list_heb += NumberToHebrew(part1)
        list_heb.append("עַד")
        list_heb += NumberToHebrew(part2)
        return list_heb

    ##    'בְּ-100,000'   / בְּ-99.99%  / הַ-1,100  /   מִ-0.7%   /  לְ-1.9    /  כְּ-22,000.
    elif is_pattern_number_with_heb(word):
        heb_letter,num,func=(clean_pattern_number_with_heb(word))
        #arr_attr= (clean_pattern_number_with_heb(word))
        list_heb = []
        list_heb.append(heb_letter)

        if func=="is_number_with_comma":
            list_heb+=NumberToHebrew(int(num))
            return list_heb

        elif func=="is_number_with_decimal":
            part1, part2 = num
            list_heb += NumberToHebrew(part1)
            list_heb += NumberToHebrew('.')
            list_heb += NumberToHebrew(part2)
            return list_heb

        elif func=="is_percentage":
            part1, part2 = clean_decimal(num)

            if part2 != None:
                list_heb += NumberToHebrew(part1)
                list_heb += NumberToHebrew('.')
                list_heb += NumberToHebrew(part2)
                list_heb += NumberToHebrew('%')
                return list_heb
            else:
                list_heb += NumberToHebrew(part1)
                list_heb += NumberToHebrew('%')
                return list_heb

        elif func == "is_number_range":
            part1, part2 = num
            list_heb += NumberToHebrew(int(part1))
            list_heb.append("עַד")
            list_heb += NumberToHebrew(int(part2))
            return list_heb

#######################################################

##takes a letter in hebrew and returns the sound in english
def HebrewLetterToEnglishSound(obj,tzuptzik,last_letter=False):
    obj = Hebrew(obj).string
    # map the nikud symbols to their corresponding phenoms
    nikud_map = {"ָ": "a", "ַ": "a", "ֶ": "e", "ֵ": "e", "ִ": "i", "ְ": "", "ֹ": "o", "ֻ": "oo", 'ּ': "", 'ֲ': 'a'}


    beged_kefet_shin_sin = {
        ############ B
        "בּ": "b",
        "בְּ": "b",
        "בִּ": "bi",
        "בֹּ": "bo",
        "בֵּ": "be",
        "בֶּ": "be",
        "בַּ": "ba",
        "בָּ": "ba",
        "בֻּ": "boo",
        ############ G
        "גּ": "g",
        "גְּ": "g",
        "גִּ": "gi",
        "גֹּ": "go",
        "גֵּ": "ge",
        "גֶּ": "ge",
        "גַּ": "ga",
        "גָּ": "ga",
        "גֻּ": "goo",
        ########### D
        "דּ": "d",
        "דְּ": "d",
        "דִּ": "di",
        "דֹּ": "do",
        "דֵּ": "de",
        "דֶּ": "de",
        "דַּ": "da",
        "דָּ": "da",
        "דֻּ": "doo",
        ########### K
        "כּ": "k",
        "כְּ": "k",
        "כִּ": "ki",
        "כֹּ": "ko",
        "כֵּ": "ke",
        "כֶּ": "ke",
        "כַּ": "ka",
        "כָּ": "ka",
        "כֻּ": "koo",
        ############ P
        "פּ": "p",
        "פְּ": "p",
        "פִּ": "pi",
        "פֹּ": "po",
        "פֵּ": "pe",
        "פֶּ": "pe",
        "פַּ": "pa",
        "פָּ": "pa",
        "פֻּ": "poo",
        ############ T
        "תּ": "t",
        "תְּ": "t",
        "תִּ": "ti",
        "תֹּ": "to",
        "תֵּ": "te",
        "תֶּ": "te",
        "תַּ": "ta",
        "תָּ": "ta",
        "תֻּ": "too",
        ############ S
        "שׂ": "s",
        "שְׂ": "s",
        "שִׂ": "si",
        "שֹׂ": "so",
        "שֵׂ": "se",
        "שֶׂ": "se",
        "שַׂ": "sa",
        "שָׂ": "sa",
        "שֻׂ": "soo",
        ########### SH
        "שׁ": "sh",
        "שְׁ": "sh",
        "שִׁ": "shi",
        "שֹׁ": "sho",
        "שֵׁ": "she",
        "שֶׁ": "she",
        "שַׁ": "sha",
        "שָׁ": "sha",
        "שֻׁ": "shoo",
    }

    vav = {
        "וֵּו": "ve",
        "וּ": "oo",
        "וּו": "oo",
        "וֹ": "o",
        "וֹו": "oo",
        "וְ": "ve",
        "וֱו": "ve",
        "וִ": "vi",
        "וִו": "vi",
        "וַ": "va",
        "וַו": "va",
        "וֶ": "ve",
        "וֶו": "ve",
        "וָ": "va",
        "וָו": "va",
        "וֻ": "oo",
        "וֻו": "oo"
    }


    letters_map = {
        "א": "",
        "ב": "v",
        "ג": "g",
        "ד": "d",
        "ה": "hh",
        "ו": "v",
        "ז": "z",
        "ח": "h",
        "ט": "t",
        "י": "y",
        "כ": "h",
        "ל": "l",
        "מ": "m",
        "נ": "n",
        "ס": "s",
        "ע": "",
        "פ": "f",
        "צ": "ts",
        "ק": "k",
        "ר": "r",
        "ש": "sh",
        "ת": "t",
        "ן": "n",
        "ם": "m",
        "ף": "f",
        "ץ": "ts",
        "ך": "h",
    }

    patah_ganav={
        "חַ": "ah",
        "חָ": "ah",
        "הַ": "hha",
        "הָ": "hha",
        "עַ": "a",
        "עָ": "a",

    }

    tzuptzik_letters={
        ##G
        "ג": "j",
        "גְ": "j",
        "גִ": "ji",
        "גֹ": "jo",
        "גֵ": "je",
        "גֶ": "je",
        "גַ": "ja",
        "גָ": "ja",
        "גֻ": "joo",
        "גּ": "j",
        "גְּ": "j",
        "גִּ": "ji",
        "גֹּ": "jo",
        "גֵּ": "je",
        "גֶּ": "je",
        "גַּ": "ja",
        "גָּ": "ja",
        "גֻּ": "joo",

        ##ch
        "צ": "ch",
        "צְ": "ch",
        "צִ": "chi",
        "צֹ": "cho",
        "צֵ": "che",
        "צֶ": "che",
        "צַ": "cha",
        "צָ": "cha",
        "צֻ": "choo",

        ##ch
        "ץ": "ch",
        "ץְ": "ch",
        "ץִ": "chi",
        "ץֹ": "cho",
        "ץֵ": "che",
        "ץֶ": "che",
        "ץַ": "cha",
        "ץָ": "cha",
        "ץֻ": "choo",

        ##Z
        "ז": "zh",
        "זְ": "zh",
        "זִ": "zhi",
        "זֹ": "zho",
        "זֵ": "zhe",
        "זֶ": "zhe",
        "זַ": "zha",
        "זָ": "zha",
        "זֻ": "zhoo",
    }

    if last_letter:
        if obj in patah_ganav:
            return patah_ganav[obj]

    if tzuptzik==True:
        if obj in tzuptzik_letters:
            return tzuptzik_letters[obj]

    if obj in beged_kefet_shin_sin:
        return beged_kefet_shin_sin[obj]
    elif obj in vav:
        return vav[obj]
    else:
        lst = break_to_list(obj)
        string = ""
        for item in lst:
            if item in letters_map:
                string += letters_map[item]
            if item in nikud_map:
                string += nikud_map[item]

        return string


##takes hebrew word and turns it into the sound in english
def HebrewWordToEnglishSound(word,index):
    new_sentence=""
    hs = Hebrew(word)
    hs = Hebrew(list(hs.graphemes)).string
    for i, letter in enumerate(hs):

        tzuptzik = False
        if i < len(hs) - 1:
            if hs[i + 1] == '\'':
                tzuptzik = True

        tav = HebrewLetterToEnglishSound(letter, tzuptzik, i == len(hs) - 1)
        new_sentence += tav

    ##clean list:
    try:
        if new_sentence[-1] == 'y' and new_sentence[-2] == 'y':
            new_sentence = new_sentence.replace("yy", "y")
    except:
        pass
    return new_sentence

##takes hebrew sentence and turns it into english sounds
def HebrewToEnglish(sentence,index=0):
    words = sentence.split()
    new_sentence = ""

    for word in words:
        ##if number not in string
        if not has_number(word):

            ##breaks the word to letters and ',' and '.'
            broken_word=break_to_letter_and_rebuild(word)

            for brk_word in broken_word:

                ##tries to add silence
                if brk_word=='.' or brk_word==',' or brk_word==';':
                    new_sentence += "q"+" "

                else:
                    ret_sentence=HebrewWordToEnglishSound(brk_word,index)
                    new_sentence+=ret_sentence+" "

        ##if there is a number:
        else:
            try:
                before_num,num,after_num=split_number_and_string(word)

                if has_number(after_num) or has_number(before_num):
                    list_of_numbers=clean_number(word)
                    for number in list_of_numbers:
                        ret_sentence = HebrewWordToEnglishSound(number, index)
                        new_sentence += ret_sentence + " "

                else:
                    ret_sentence = HebrewWordToEnglishSound(before_num, index)
                    new_sentence += ret_sentence+" "

                    num = [s for s in word if s.isdigit()]
                    num="".join(num)
                    num=int(num)
                    list_of_numbers=NumberToHebrew(num)
                    for number in list_of_numbers:
                        ret_sentence=HebrewWordToEnglishSound(number,index)
                        new_sentence += ret_sentence + " "

                    ret_sentence = HebrewWordToEnglishSound(after_num, index)
                    new_sentence += ret_sentence + " "



            except:
                print("error from split_number_and_string in line:", index,"with word: ",word)



    return new_sentence
