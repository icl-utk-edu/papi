int
main (int argc, char *argv[]) {
#if defined(__i386__) || defined(i386) || defined(_M_IX86)
    return 1;
#endif
    return 0;
}
