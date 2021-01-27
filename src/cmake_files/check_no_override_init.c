struct A {int x; int y;};
    
int
main(int argc, char *argv[])
{
    struct A a = {.x = 0, .y = 0, .y = 5};
    return a.x;
}
