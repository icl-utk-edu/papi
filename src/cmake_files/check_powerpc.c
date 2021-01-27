int main (int argc, char *argv[]) {
#if defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC) || defined(__powerpc64__)
    return 1;
#endif
    return 0;
}
