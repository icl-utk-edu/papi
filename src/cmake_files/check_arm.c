int main (int argc, char *argv[]) {
#if defined(__arm__) || defined(_M_ARM) || defined(__arm)
    return 1;
#endif
    return 0;
}
