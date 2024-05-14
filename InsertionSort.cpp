#include<iostream>
using namespace std;

void Insertion_Sort(int arr[],int size){
    for(int i=1;i<size;i++){
        int element =arr[i];
        int j=i-1;
        for(;j>=0;j--){
            if(arr[j]>element){     //shift
                arr[j+1]=arr[j];
            }
            else{
                break;              // no need to compare from previous 
            }
        }
        arr[j+1]=element;
    }

}

int main(){
    int n=9;
    int array[]={2,1,4,6,3,7,1,9,6};
    Insertion_Sort(array,n);
    for(auto x:array){
        cout<<x<<endl;
    }
return 0;
}