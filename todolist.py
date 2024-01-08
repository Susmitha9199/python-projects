tasks=[]
def addTask():
    task=input("please enter a task:")
    tasks.append(task)
    print(f"Task '{task}' added to list.")
def listTasks():
    if not tasks:
        print("there are not task currently")
    else:
        print("current task:")
        for index, task in enumerate(tasks):
            print(f"task #{index}.{task} ")
def deleteTask():
    listTasks()
    try:
        tasktodelete= int(input("Chosse the number that you want to delete:"))
        if tasktodelete>=0 and tasktodelete < len(tasks):
            tasks.pop(tasktodelete)
            print(f"Task '{tasktodelete}' has been delted")
        else:
            print(f"Task '{tasktodelete}' was not found in the list.")
    except:
        print("Invaild input")
    
if __name__ == "__main__":
    print("welcome to the to do list app:)")
    while True:
        print("\n")
        print("please select one of the following options")
        print("------------------------------------------")
        print("1. Add a new task")
        print("2. Delete a task")
        print("3. List task")
        print("4. Quit")
        choice=int(input("enter your choice : "))
        if (choice==1):
            addTask()
        elif(choice==2):
            deleteTask()
        elif(choice==3):
            listTasks()
        elif(choice==4):
            break
        else:
            print("Invalid input :(, Please try again")
    print("God bye :) ")
    
        