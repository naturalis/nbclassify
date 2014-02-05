import os

pictures = open("pictures.txt", 'r').readlines()
for picture in pictures:
    no_enter = picture.strip()
    name = no_enter.split(".jpg")[0]
    #print ID + "_tags.txt"
    tags = name + "_tags.txt"
    tag = open(tags, 'r').readlines()
    print "convert %s to %s.png" %(no_enter, name)
    os.system("convert %s %s.png" %(no_enter, name))
    if "Look-a-Like_round\n" in tag:
        os.system("mv %s.png LRound/%s.png" %(name, name))
        os.system("mv %s LRound/%s" %(tags, tags))
    elif "Look-a-Like_oblong\n" in tag:
        os.system("mv %s.png LOblong/%s.png" %(name, name))
        os.system("mv %s LOblong/%s" %(tags, tags))
    elif "Look-a-Like_spur\n" in tag:
        os.system("mv %s.png LSpur/%s.png" %(name, name))
        os.system("mv %s LSpur/%s" %(tags, tags))
    elif "Round\n" in tag:
        os.system("mv %s.png Round/%s.png" %(name, name))
        os.system("mv %s Round/%s" %(tags, tags))
    elif "Oblong\n" in tag:
        os.system("mv %s.png Oblong/%s.png" %(name, name))
        os.system("mv %s Oblong/%s" %(tags, tags))
    elif "Spur\n" in tag:
        os.system("mv %s.png Spur/%s.png" %(name, name))
        os.system("mv %s Spur/%s" %(tags, tags))
    else:
        tag.close()
        print y, "NO"
