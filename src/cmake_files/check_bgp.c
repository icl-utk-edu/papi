int main (int argc, char *argv[]) {
#if defined(__bgp__)
    return 1;
#endif
    return 0;
}
