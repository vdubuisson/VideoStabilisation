def create_stabilized_path(path, extension):
    
    path_stabilized=""
    portion_indice=0
    for portion in path.split('.')[0:len(path.split('.'))-1]:
        if not portion:
            portion="."
        if portion_indice== len(path.split('.'))-2:
            portion=portion+"_stabilized"
        path_stabilized+=portion
        portion_indice+=1

    path_stabilized=path_stabilized+extension
    return path_stabilized

def border_management(arg):

    if len(arg)<3:
        border_type="black"
    else:
        border_type=arg[2]
    return border_type
