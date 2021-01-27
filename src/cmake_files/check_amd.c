int main (int argc, char *argv[]) {
#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(_M_AMD64)
    return 1;
#endif
    return 0;
}
