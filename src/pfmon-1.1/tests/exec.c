#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

extern char **environ;
int
main(int argc, char **argv)
{
	pid_t pid;

	if (argc < 2) exit(1);

	switch((pid=fork())) {
	case -1: perror("fork"); exit(1);
	case  0: return execve(argv[1], argv+1, environ);
	}
printf("parent wait\n");
	waitpid(pid, NULL, 0); 
printf("parent exit\n");
	return 0;
}

