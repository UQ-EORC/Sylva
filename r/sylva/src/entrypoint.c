// Registers the Rust routines with R.
void R_init_sylva_extendr(void *dll);

void R_init_sylva(void *dll) {
    R_init_sylva_extendr(dll);
}
